import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

# 利用Kalman滤波器融合AutoTrack跟踪结果(遮挡Kalman预测占比较大）

class AutoTrack:
    def __init__(self):
        # ================= 超参数 =================
        self.padding = 1
        self.lambda_reg = 1e-2

        # 时间正则（AutoTrack）
        self.ref_mu = 0
        self.epsilon = 1
        self.mu = 0
        self.delta = 0.2 #响应变化调整系数
        self.phi = 0.3 #更新阈值

        #更新ref_mu
        self.zeta = 20.0
        self.nu = 0.2

        self.eta = 0
        self.psr = 0

        self.theta_max = 18.0
        self.theta_min = 2.0
        self.alpha1 = 0.05

        self.alpha = 0.2
        self.beta = 0.2

        # ADMM
        self.gamma_init = 1.0
        self.gamma_max = 10000.0
        self.gamma_step = 10.0
        self.admm_iter = 4

        self.output_sigma_factor = 0.1

        # HOG
        self.hog_cell_size = 4
        self.hog_nbins = 9
        self.hog = None

        # ================= 状态 =================
        self.pos = None
        self.target_sz = None
        self.window_sz = None

        self.yf = None
        self.reg_window = None
        self.reg_window1 = None
        self.reg_min = 1e-3
        self.reg_max = 1e5

        self.Hc, self.Wc = None, None

        # AutoTrack变量
        self.g_f = None      # 主滤波器
        self.h_f = None      # 空间正则变量
        self.l_f = None      # 拉格朗日乘子
        self.g_pre = None

        self.cos_window = None
        self.response_prev = None
        self.disp_prev = None

        # K帧量
        self.K = 3 #所取帧数
        self.resp_queue = deque(maxlen=self.K) #响应图队列（存储K帧)
        self.disp_queue = deque(maxlen=self.K) #位置队列（存储K帧），用于记录每一帧响应图峰值的
        self.psi_queue = deque(maxlen=self.K) #系数队列（存储K帧）

        # ================= 卡尔曼滤波器（用于位置修正）=================
        # 状态向量: [cx, cy, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # 状态转移矩阵 A (匀速运动模型)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ]
        ], dtype=np.float32)
        
        # 观测矩阵 H
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差矩阵 Q
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # 观测噪声协方差矩阵 R (根据PSR动态调整)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        # 后验误差协方差矩阵 P
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # 卡尔曼融合权重参数
        self.kalman_weight_high_psr = 0.5  # PSR高时，30%卡尔曼，70%检测
        self.kalman_weight_low_psr = 0.8   # PSR低时，70%卡尔曼，30%检测
        self.psr_smooth_threshold = 8.0    # PSR平滑阈值


    # ==================================================
    # 初始化
    # ==================================================
    def init(self, frame, bbox):
        x, y, w, h = bbox
        w = w // 4 * 4
        h = h // 4 * 4

        self.pos = np.array([y + h / 2, x + w / 2], dtype=np.float32)
        self.target_sz = np.array([h, w], dtype=np.float32)
        self.window_sz = np.floor(self.target_sz * (1 + self.padding)).astype(int)

        cs = self.hog_cell_size
        Hc = self.window_sz[0] // cs
        Wc = self.window_sz[1] // cs
        self.Hc, self.Wc = Hc, Wc
        self.window_sz = [Hc * cs, Wc * cs]

        self.cos_window = np.outer(np.hanning(Hc), np.hanning(Wc))

        output_sigma = np.sqrt(np.prod(self.target_sz)) * self.output_sigma_factor / cs
        y = self._gaussian_label((Hc, Wc), output_sigma)
        self.yf = np.fft.fft2(y)

        self.reg_window = self._init_reg_window((Hc, Wc))
        self.reg_window1 = self.reg_window

        self.hog = cv2.HOGDescriptor(
            (self.window_sz[1], self.window_sz[0]),
            (cs, cs), (cs, cs), (cs, cs),
            self.hog_nbins
        )

        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))

        self.g_f = np.zeros_like(xf)
        self.h_f = np.zeros_like(xf)
        self.l_f = np.zeros_like(xf)
        self.g_pre = np.zeros_like(xf)

        self._train(frame)

        disp, response = self._detect(frame)
        self.response_prev = response
        self.disp_prev = disp

        # 初始化卡尔曼滤波器状态
        self.kf.statePost = np.array([
            [self.pos[1]],  # cx
            [self.pos[0]],  # cy
            [0],            # vx
            [0]             # vy
        ], dtype=np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        
        print("AutoTrack initialized with Kalman position correction")
        print(f"Initial position: ({self.pos[1]:.2f}, {self.pos[0]:.2f})")

    # ==================================================
    # 跟踪（保持原始逻辑 + 卡尔曼位置修正）
    # ==================================================
    def track(self, frame):
        """
        跟踪流程（保持原始AutoTrack_improved2逻辑）：
        1. 相关滤波检测
        2. 卡尔曼预测和位置融合（修正）
        3. 根据PSR更新ref_mu
        4. 训练相关滤波器（按原始逻辑）
        5. 更新卡尔曼滤波器
        """
        
        # ========== Step 1: 卡尔曼预测 ==========
        prediction = self.kf.predict()
        pred_cx = prediction[0, 0]
        pred_cy = prediction[1, 0]
        pred_vx = prediction[2, 0]
        pred_vy = prediction[3, 0]
        predicted_pos = np.array([pred_cy, pred_cx], dtype=np.float32)
        
        # ========== Step 2: 相关滤波检测 ==========
        disp, response = self._detect(frame)
        detected_pos = self.pos + disp
        
        # ========== Step 3: 计算PSR ==========
        psr = self.compute_psr(response)
        
        # ========== Step 4: 卡尔曼位置融合（修正检测位置）==========
        # 根据PSR自适应融合卡尔曼预测和相关滤波检测
        fusion_weight = self._compute_fusion_weight(psr)
        corrected_pos = fusion_weight * predicted_pos + (1 - fusion_weight) * detected_pos
        
        # 更新位置为融合后的位置
        self.pos = corrected_pos
        
        # ========== Step 5: 更新卡尔曼滤波器（用融合后的位置）==========
        measurement = np.array([
            [self.pos[1]],  # cx
            [self.pos[0]]   # cy
        ], dtype=np.float32)
        
        # 根据PSR调整观测噪声（需要拷贝赋值避免类型不匹配）
        obs_noise = self._adaptive_observation_noise(psr)
        self.kf.measurementNoiseCov[:] = obs_noise
        self.kf.correct(measurement)
        
        # ========== Step 6: 更新ref_mu（原始逻辑）==========
        occ = False
        if response is not None and self.response_prev is not None:
            response_diff = self._align_and_diff_response(response, disp)
            ref_mu, occ = self._update_ref_mu1(response, psr, self.theta_max, self.theta_min)
            self.ref_mu = ref_mu
            self.mu = self.zeta
            
            # 更新正则化窗口
            delta_resp = self.delta * np.log(1 + response_diff)
            delta_resp = cv2.blur(delta_resp, (3, 3))
            self.reg_window1 = np.clip(self.reg_window + delta_resp, self.reg_min, self.reg_max)
        
        # ========== Step 7: 训练相关滤波器（原始逻辑：未遮挡时训练）==========
        if occ == False:
            # 使用融合后的位置进行训练
            self._train(frame, response, disp)
        
        # ========== Step 8: 更新历史队列（原始逻辑）==========
        self.resp_queue.appendleft(response)
        self.disp_queue.appendleft(disp)
        psi = self.compute_psi_from_psr(self.psr)
        self.psi_queue.appendleft(psi)
        
        self.response_prev = response
        self.disp_prev = disp
        
        # 打印调试信息
        print(f"PSR: {psr:.2f} | Detected: ({detected_pos[1]:.1f}, {detected_pos[0]:.1f}) | "
              f"Kalman: ({predicted_pos[1]:.1f}, {predicted_pos[0]:.1f}) | "
              f"Fused: ({self.pos[1]:.1f}, {self.pos[0]:.1f}) | Weight: {fusion_weight:.2f} | "
              f"Occ: {occ}")

        return self.pos.copy(), self.target_sz.copy(), response

    # ==================================================
    # 检测
    # ==================================================
    def _detect(self, frame):
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))

        response_f = np.sum(np.conj(self.g_f) * xf, axis=2)
        response = np.real(np.fft.ifft2(response_f))

        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        dy -= response.shape[0] // 2
        dx -= response.shape[1] // 2

        disp = np.array([dy, dx], dtype=np.float32) * self.hog_cell_size
        return disp, response

    # ==================================================
    # AutoTrack ADMM 训练
    # ==================================================
    def _train(self, frame, response=None, disp=None):
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))

        yf = self.yf
        T = xf.shape[0] * xf.shape[1]

        S_xx = np.sum(np.conj(xf) * xf, axis=2)
        Sgx_pre = np.sum(np.conj(xf) * self.g_pre, axis=2)

        mu = self.mu
        self.g_f = np.zeros_like(self.g_pre)
        self.h_f = np.zeros_like(self.g_pre)
        self.l_f = np.zeros_like(self.g_pre)
        gamma = self.gamma_init

        alpha = 0
        R = 0
        #增加对前K帧的处理
        if len(self.resp_queue) == self.K and disp is not None:
            dy0, dx0 = int(disp[0]/ self.hog_cell_size), int(disp[1]/ self.hog_cell_size)
            for Rk, dispk, psik in zip(self.resp_queue, self.disp_queue, self.psi_queue):
                alpha += psik
                dyk, dxk = int(dispk[0]/ self.hog_cell_size), int(dispk[1]/ self.hog_cell_size)
                dy = dy0 - dyk
                dx = dx0 - dxk
                Rk0 = np.roll(Rk, shift=(dy, dx), axis=(0, 1))
                R += psik * Rk0
                alpha += psik


        for _ in range(self.admm_iter):
            # ===== g 子问题 =====
            B = S_xx * (1 + alpha)+ T * (gamma + mu)


            Shx = np.sum(np.conj(xf) * self.h_f, axis=2)
            Slx = np.sum(np.conj(xf) * self.l_f, axis=2)

            term1 = ((yf + R)[..., None] * xf) / (T * (gamma + mu))
            term2 = - self.l_f / (gamma + mu)
            term3 = (gamma * self.h_f) / (gamma + mu)
            term4 = (mu * self.g_pre) / (gamma + mu)


            corr = (
                (xf * (S_xx * yf)[..., None]) / (T * (gamma + mu))
                + (mu * xf * Sgx_pre[..., None]) / (gamma + mu)
                - (xf * Slx[..., None]) / (gamma + mu)
                + (gamma * xf * Shx[..., None]) / (gamma + mu)
                + (xf * (S_xx * R)[..., None]) / (gamma + mu)
            )

            self.g_f = term1 + term2 + term3 + term4 - corr / B[..., None]


            # ===== h 子问题（空间正则）=====
            denom = self.lambda_reg * self.reg_window1[..., None] ** 2 + gamma * T
            lhd = T / denom
            X = np.real(np.fft.ifft2(gamma * (self.g_f + self.l_f), axes=(0, 1)))
            self.h_f = np.fft.fft2(lhd * X, axes=(0, 1))

            # ===== mu 更新（时间正则）=====
            diff = np.sum(np.abs(self.g_f - self.g_pre) ** 2)
            z = diff / (2 * self.epsilon)
            mu = self.ref_mu - z

            # ===== 拉格朗日乘子 =====
            self.l_f += gamma * (self.g_f - self.h_f)
            gamma = min(gamma * self.gamma_step, self.gamma_max)

        self.g_pre = self.g_f

    # ==================================================
    # 特征
    # ==================================================
    def _get_features(self, frame, pos):
        y, x = pos.astype(int)
        h, w = self.window_sz

        y1, y2 = y - h // 2, y - h // 2 + h
        x1, x2 = x - w // 2, x - w // 2 + w

        y1c, y2c = max(0, y1), min(frame.shape[0], y2)
        x1c, x2c = max(0, x1), min(frame.shape[1], x2)

        patch = frame[y1c:y2c, x1c:x2c]
        patch = cv2.copyMakeBorder(
            patch,
            y1c - y1, y2 - y2c,
            x1c - x1, x2 - x2c,
            cv2.BORDER_REPLICATE
        )

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hog_feat = self.hog.compute(gray)
        hog_feat = hog_feat.reshape(self.Wc, self.Hc, self.hog_nbins).transpose(1, 0, 2)
        hog_feat = hog_feat.astype(np.float32)

        hog_feat -= hog_feat.mean(axis=(0, 1), keepdims=True)
        hog_feat *= self.cos_window[..., None]
        return hog_feat

    # ==================================================
    # 对齐
    # ==================================================
    def _align_and_diff_response(self, response, disp):
        dy, dx = int(disp[0] / self.hog_cell_size), int(disp[1] / self.hog_cell_size)
        dy_p, dx_p = int(self.disp_prev[0] / self.hog_cell_size), int(self.disp_prev[1] / self.hog_cell_size)

        r1 = np.roll(response, (-dy, -dx), axis=(0, 1))
        r2 = np.roll(self.response_prev, (-dy_p, -dx_p), axis=(0, 1))

        return np.abs(np.abs(r1) - np.abs(r2)) / (np.abs(r2) + 1e-6)

    def _update_ref_mu(self, response_diff):
        m = self.zeta
        p = self.nu
        eta = np.linalg.norm(response_diff.ravel(), 2) / 1e4
        self.eta = eta

        if eta < self.phi:
            return m / (1 + np.log(p * eta + 1)), False
        else:
            return 50.0, True

    def _update_ref_mu1(self, response, psr, theta_max, theta_min):
        psr_value = 8
        alpha = self.alpha1
        if psr > psr_value:
            ref_mu = theta_min + (theta_max - theta_min) * np.exp(-alpha * (psr-psr_value))
            return ref_mu, False
        else:
            return theta_max, True

    def compute_psr(self, response, peak_size=5):
        """计算PSR"""
        H, W = response.shape
        py, px = np.unravel_index(np.argmax(response), response.shape)

        half = peak_size // 2
        y1, y2 = max(0, py - half), min(H, py + half + 1)
        x1, x2 = max(0, px - half), min(W, px + half + 1)

        side = response.copy()
        side[y1:y2, x1:x2] = 0

        mean = np.mean(side)
        std = np.std(side) + 1e-6

        psr = (response[py, px] - mean) / std
        self.psr = psr
        return psr

    def compute_psi_from_psr(self, psr):
        """根据PSR计算权重系数"""
        alpha = self.alpha
        beta = self.beta
        psi = 1/(1 + np.log(1 + alpha * psr)) - beta
        return psi

    # ==================================================
    # 卡尔曼相关方法
    # ==================================================
    def _compute_fusion_weight(self, psr):
        """
        计算卡尔曼预测和相关滤波检测的融合权重
        返回卡尔曼的权重（0-1之间）
        PSR越高，越信任相关滤波检测（卡尔曼权重越低）
        PSR越低，越信任卡尔曼预测（卡尔曼权重越高）
        """
        if psr > self.psr_smooth_threshold:
            # PSR高，信任检测
            return self.kalman_weight_high_psr
        else:
            # PSR低，增加卡尔曼权重（线性插值）
            # PSR从0到threshold，权重从low到high
            ratio = max(0, min(1, psr / self.psr_smooth_threshold))
            weight = self.kalman_weight_low_psr + (self.kalman_weight_high_psr - self.kalman_weight_low_psr) * ratio
            return weight

    def _adaptive_observation_noise(self, psr):
        """根据PSR自适应调整观测噪声"""
        min_noise = 0.1   # PSR很高时
        max_noise = 5.0   # PSR很低时
        
        if psr > self.psr_smooth_threshold:
            noise = min_noise
        else:
            # PSR越低，噪声越大（越不信任观测）
            noise = max_noise - (max_noise - min_noise) * (psr / self.psr_smooth_threshold)
        
        # 确保返回的矩阵类型为 float32
        return (np.eye(2, dtype=np.float32) * np.float32(noise)).astype(np.float32)

    # ==================================================
    # 工具
    # ==================================================
    def _gaussian_label(self, sz, sigma):
        h, w = sz
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        return np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2) / sigma ** 2)

    def _init_reg_window(self, sz):
        h, w = sz
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        dist = (y - cy) ** 2 + (x - cx) ** 2
        return np.exp(dist / (0.5 * h * w))


if __name__ == '__main__':

    ax_flag = True

    cap = cv2.VideoCapture("./video/2.mp4")
    #cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    bbox = cv2.selectROI("AutoTrack Init", frame, False, False)
    cv2.destroyWindow("AutoTrack Init")

    tracker = AutoTrack()
    tracker.init(frame, bbox)

    frame_idx = 0

    if ax_flag:
        plt.ion()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # --- 左上：PSR ---
        ax1.set_title("PSR over Time")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("PSR")
        ax1.axhline(y=tracker.psr_smooth_threshold, color='g', linestyle='--', label='PSR Threshold')
        ax1.legend()
        psr_x, psr_y = [], []
        line_psr, = ax1.plot([], [], 'b-', lw=2)

        # --- 右上：卡尔曼融合权重 ---
        ax2.set_title("Kalman Fusion Weight")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Kalman Weight")
        ax2.set_ylim(0, 1)
        weight_x, weight_y = [], []
        line_weight, = ax2.plot([], [], 'r-', lw=2)

        # --- 左下：速度 ---
        ax3.set_title("Velocity Magnitude")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Speed (pixels/frame)")
        vel_x, vel_y = [], []
        line_vel, = ax3.plot([], [], 'g-', lw=2)

        # --- 右下：response heatmap ---
        ax4.set_title("Response Heatmap")
        heatmap = ax4.imshow(
            np.zeros((tracker.Hc, tracker.Wc)),
            cmap="jet",
            interpolation="nearest",
            origin="upper"
        )
        plt.colorbar(heatmap, ax=ax4)

        plt.tight_layout()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos, target_sz, response = tracker.track(frame)
        cy, cx = pos.astype(int)
        h, w = target_sz.astype(int)

        frame_idx += 1

        # 获取卡尔曼状态
        psr = tracker.psr
        vx = tracker.kf.statePost[2, 0]
        vy = tracker.kf.statePost[3, 0]
        vmag = np.sqrt(vx**2 + vy**2)
        kalman_weight = tracker._compute_fusion_weight(psr)

        if ax_flag:
            # ===== PSR 更新 =====
            psr_x.append(frame_idx)
            psr_y.append(psr)
            line_psr.set_data(psr_x, psr_y)
            ax1.relim()
            ax1.autoscale_view()

            # ===== 融合权重 更新 =====
            weight_x.append(frame_idx)
            weight_y.append(kalman_weight)
            line_weight.set_data(weight_x, weight_y)
            ax2.relim()
            ax2.autoscale_view()

            # ===== 速度 更新 =====
            vel_x.append(frame_idx)
            vel_y.append(vmag)
            line_vel.set_data(vel_x, vel_y)
            ax3.relim()
            ax3.autoscale_view()

            # ===== response heatmap 更新 =====
            heatmap.set_data(response)
            heatmap.set_clim(vmin=response.min(), vmax=response.max())

            plt.pause(0.001)

        # 绘制跟踪框（颜色根据PSR变化）
        if psr > tracker.psr_smooth_threshold:
            color = (0, 255, 0)  # 绿色：高PSR，主要信任检测
        else:
            # 黄色到红色：PSR降低，卡尔曼权重增加
            ratio = psr / tracker.psr_smooth_threshold
            color = (0, int(255 * ratio), int(255 * (1 - ratio)))
        
        cv2.rectangle(
            frame,
            (cx - w // 2, cy - h // 2),
            (cx + w // 2, cy + h // 2),
            color, 2
        )

        # 显示状态信息
        cv2.putText(frame, f"Frame: {frame_idx} | PSR: {psr:.2f}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f"Kalman Weight: {kalman_weight:.2f} | Velocity: ({vx:.1f}, {vy:.1f})", 
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("AutoTrack + Kalman Position Correction", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if ax_flag:
        plt.ioff()
        plt.show()
