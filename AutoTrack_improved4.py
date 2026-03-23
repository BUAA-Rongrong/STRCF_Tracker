import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

# AutoTrack4: 结合尺度滤波器 + Kalman位置修正
# 基于 AutoTrack_improved2_scale.py 和 AutoTrack_improved_Kalman.py 合并

class AutoTrack4:
    def __init__(self):
        # ================= 超参数 =================
        self.padding = 1
        self.lambda_reg = 1e-2

        # 时间正则（AutoTrack）
        self.ref_mu = 0
        self.epsilon = 1
        self.mu = 0
        self.delta = 0.1  # 响应变化调整系数
        self.phi = 0.3    # 更新阈值

        # 更新ref_mu
        self.zeta = 15.0
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

        # ================= 尺度参数 =================
        self.num_scales = 33
        self.scale_step = 1.03
        self.scale_sigma_factor = 0.5
        self.scale_lambda = 1e-2
        self.learning_rate_scale = 0.5
        self.scale_model_factor = 1.0
        self.scale_model_max_area = np.array([32, 32], dtype=np.float32)

        # 尺度滤波器状态
        self.scale_factors = None
        self.scale_window = None
        self.sf_num = None
        self.sf_den = None
        self.current_scale_factor = 1.0
        self.min_scale_factor = None
        self.max_scale_factor = None
        self.ysf = None
        self.scale_initialized = False

        # 基础目标尺寸
        self.base_target_sz = None
        self.im_shape = None
        self.scale_model_sz = None

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
        self.K = 3
        self.resp_queue = deque(maxlen=self.K)
        self.disp_queue = deque(maxlen=self.K)
        self.psi_queue = deque(maxlen=self.K)

        # ================= 卡尔曼滤波器（用于位置修正）=================
        self.kf = cv2.KalmanFilter(4, 2)

        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # 卡尔曼融合权重参数
        self.kalman_weight_high_psr = 0.0
        self.kalman_weight_low_psr = 0.0
        self.psr_smooth_threshold = 6.0

    # ==================================================
    # 初始化
    # ==================================================
    def init(self, frame, bbox):
        x, y, w, h = bbox
        w = w // 4 * 4
        h = h // 4 * 4

        self.pos = np.array([y + h / 2, x + w / 2], dtype=np.float32)
        self.target_sz = np.array([h, w], dtype=np.float32)
        self.base_target_sz = self.target_sz.copy()
        self.im_shape = frame.shape[:2]
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

        # 初始化尺度参数
        self._init_scale_params()

        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))

        self.g_f = np.zeros_like(xf)
        self.h_f = np.zeros_like(xf)
        self.l_f = np.zeros_like(xf)
        self.g_pre = np.zeros_like(xf)

        # 初始化位置滤波器
        self._train(frame)
        # 初始化尺度滤波器
        self._update_scale_model(frame, self.pos)

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

        print("AutoTrack4 initialized with Scale Filter + Kalman Position Correction")
        print(f"Initial position: ({self.pos[1]:.2f}, {self.pos[0]:.2f})")
        print(f"Initial target size: {w}x{h}")

    # ==================================================
    # 跟踪
    # ==================================================
    def track(self, frame):
        """
        跟踪流程：
        1. 卡尔曼预测
        2. 相关滤波检测
        3. 尺度检测
        4. 卡尔曼位置融合
        5. 更新ref_mu
        6. 训练滤波器（位置+尺度）
        7. 更新卡尔曼
        """

        # ========== Step 1: 卡尔曼预测 ==========
        prediction = self.kf.predict()
        pred_cx = prediction[0, 0]
        pred_cy = prediction[1, 0]
        predicted_pos = np.array([pred_cy, pred_cx], dtype=np.float32)

        # ========== Step 2: 相关滤波检测 ==========
        disp, response = self._detect(frame)
        detected_pos = self.pos + disp

        # ========== Step 3: 尺度检测 ==========
        if self.scale_initialized:
            self._detect_scale(frame, self.pos)
        else:
            self.scale_initialized = True

        # 更新目标显示尺寸
        self.target_sz = self.base_target_sz * self.current_scale_factor

        # ========== Step 4: 计算PSR ==========
        psr = self.compute_psr(response)

        # ========== Step 5: 卡尔曼位置融合 ==========
        fusion_weight = self._compute_fusion_weight(psr)
        corrected_pos = fusion_weight * predicted_pos + (1 - fusion_weight) * detected_pos

        # 更新位置为融合后的位置
        self.pos = corrected_pos

        # ========== Step 6: 更新卡尔曼滤波器 ==========
        measurement = np.array([
            [self.pos[1]],  # cx
            [self.pos[0]]   # cy
        ], dtype=np.float32)

        obs_noise = self._adaptive_observation_noise(psr)
        self.kf.measurementNoiseCov[:] = obs_noise
        self.kf.correct(measurement)

        # ========== Step 7: 更新ref_mu ==========
        occ = False
        if response is not None and self.response_prev is not None:
            response_diff = self._align_and_diff_response(response, disp)
            ref_mu, occ = self._update_ref_mu1(response, psr, self.theta_max, self.theta_min)
            self.ref_mu = ref_mu
            self.mu = self.zeta

            delta_resp = self.delta * np.log(1 + response_diff)
            delta_resp = cv2.blur(delta_resp, (3, 3))
            self.reg_window1 = np.clip(self.reg_window + delta_resp, self.reg_min, self.reg_max)

        # ========== Step 8: 训练滤波器 ==========
        if occ == False:
            self._train(frame, response, disp)
            # 更新尺度模型
            self._update_scale_model(frame, self.pos)

        # ========== Step 9: 更新历史队列 ==========
        self.resp_queue.appendleft(response)
        self.disp_queue.appendleft(disp)
        psi = self.compute_psi_from_psr(self.psr)
        self.psi_queue.appendleft(psi)

        self.response_prev = response
        self.disp_prev = disp

        # 打印位置信息
        print(f"Detected: ({detected_pos[1]:.1f}, {detected_pos[0]:.1f}) | "
              f"Kalman: ({pred_cx:.1f}, {pred_cy:.1f}) | "
              f"Fused: ({corrected_pos[1]:.1f}, {corrected_pos[0]:.1f})")

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
        if len(self.resp_queue) == self.K and disp is not None:
            dy0, dx0 = int(disp[0] / self.hog_cell_size), int(disp[1] / self.hog_cell_size)
            for Rk, dispk, psik in zip(self.resp_queue, self.disp_queue, self.psi_queue):
                alpha += psik
                dyk, dxk = int(dispk[0] / self.hog_cell_size), int(dispk[1] / self.hog_cell_size)
                dy = dy0 - dyk
                dx = dx0 - dxk
                Rk0 = np.roll(Rk, shift=(dy, dx), axis=(0, 1))
                R += psik * Rk0
                alpha += psik

        for _ in range(self.admm_iter):
            # ===== g 子问题 =====
            B = S_xx * (1 + alpha) + T * (gamma + mu)

            Shx = np.sum(np.conj(xf) * self.h_f, axis=2)
            Slx = np.sum(np.conj(xf) * self.l_f, axis=2)

            term1 = ((yf + R)[..., None] * xf) / (T * (gamma + mu))
            term2 = -self.l_f / (gamma + mu)
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

        gray = gray.astype(np.float32) / 255.0
        gray_resized = cv2.resize(gray, (self.Wc, self.Hc))
        gray_feat = gray_resized[..., None]
        hog_feat = np.concatenate((hog_feat, gray_feat), axis=2)

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
        psr_value = 8.0
        alpha = self.alpha1
        if psr > psr_value:
            ref_mu = theta_min + (theta_max - theta_min) * np.exp(-alpha * (psr - psr_value))
            return ref_mu, False
        else:
            return theta_max, True

    def compute_psr(self, response, peak_size=5):
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
        alpha = self.alpha
        beta = self.beta
        psi = 1 / (1 + np.log(1 + alpha * psr)) - beta
        return psi

    # ==================================================
    # 卡尔曼相关方法
    # ==================================================
    def _compute_fusion_weight(self, psr):
        if psr > self.psr_smooth_threshold:
            return self.kalman_weight_high_psr
        else:
            ratio = max(0, min(1, psr / self.psr_smooth_threshold))
            weight = self.kalman_weight_low_psr + (self.kalman_weight_high_psr - self.kalman_weight_low_psr) * ratio
            return weight

    def _adaptive_observation_noise(self, psr):
        min_noise = 0.1
        max_noise = 5.0

        if psr > self.psr_smooth_threshold:
            noise = min_noise
        else:
            noise = max_noise - (max_noise - min_noise) * (psr / self.psr_smooth_threshold)

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

    # ================= 尺度相关方法 =================
    def _init_scale_params(self):
        """初始化尺度参数（参考DSST）"""
        n = self.num_scales
        self.scale_factors = self.scale_step ** (np.arange(n) - n // 2)

        scale_sigma = np.sqrt(n) * self.scale_sigma_factor
        ss = np.arange(n) - n // 2
        ys = np.exp(-0.5 * (ss ** 2) / scale_sigma ** 2)
        self.ysf = np.fft.fft(ys)

        self.scale_window = np.hanning(n)

        self.scale_model_sz = (self.base_target_sz * self.scale_model_factor).astype(int)
        if self.base_target_sz[0] * self.base_target_sz[1] * self.scale_model_factor ** 2 > self.scale_model_max_area[0] * self.scale_model_max_area[1]:
            self.scale_model_sz = self.scale_model_max_area.astype(int)

        cs = self.hog_cell_size
        self.scale_hog = cv2.HOGDescriptor(
            _winSize=(self.scale_model_sz[1], self.scale_model_sz[0]),
            _blockSize=(cs, cs),
            _cellSize=(cs, cs),
            _blockStride=(cs, cs),
            _nbins=self.hog_nbins
        )

        min_scale = np.max(5.0 / self.base_target_sz)
        max_scale = np.min([self.im_shape[0], self.im_shape[1]] / self.base_target_sz)

        self.min_scale_factor = self.scale_step ** np.ceil(np.log(min_scale) / np.log(self.scale_step))
        self.max_scale_factor = self.scale_step ** np.floor(np.log(max_scale) / np.log(self.scale_step))

    def _extract_scale_sample(self, frame, pos, base_target_sz, scale_factors):
        """提取不同尺度的样本（HOG + 灰度特征）"""
        n_scales = len(scale_factors)
        cs = self.hog_cell_size

        # 计算灰度特征的尺寸（与HOG特征图空间尺寸一致）
        gray_feat_h = max(1, self.scale_model_sz[0] // cs)
        gray_feat_w = max(1, self.scale_model_sz[1] // cs)
        gray_feat_len = gray_feat_h * gray_feat_w

        for i, scale_factor in enumerate(scale_factors):
            # 计算当前尺度的目标大小
            patch_sz = np.floor(base_target_sz * scale_factor).astype(int)
            patch_sz = np.maximum(patch_sz, 10)

            # 提取图像块
            y, x = pos.astype(int)
            y1, y2 = y - patch_sz[0] // 2, y - patch_sz[0] // 2 + patch_sz[0]
            x1, x2 = x - patch_sz[1] // 2, x - patch_sz[1] // 2 + patch_sz[1]

            # 边界检查
            y1c, y2c = max(0, y1), min(frame.shape[0], y2)
            x1c, x2c = max(0, x1), min(frame.shape[1], x2)

            patch = frame[y1c:y2c, x1c:x2c]

            if patch.size == 0:
                hog_feat = np.zeros(self.hog_nbins, dtype=np.float32)
                gray_feat = np.zeros(gray_feat_len, dtype=np.float32)
            else:
                # 复制边缘
                patch = cv2.copyMakeBorder(
                    patch,
                    y1c - y1, y2 - y2c,
                    x1c - x1, x2 - x2c,
                    cv2.BORDER_REPLICATE
                )

                # resize到尺度模型固定尺寸
                patch_resized = cv2.resize(patch, (self.scale_model_sz[1], self.scale_model_sz[0]))
                gray = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)

                # 1. HOG特征
                try:
                    hog_feat = self.scale_hog.compute(gray)
                    # L2归一化
                    hog_feat /= (np.linalg.norm(hog_feat) + 1e-6)
                except:
                    hog_feat = np.zeros(self.hog_nbins, dtype=np.float32)

                # 2. 灰度特征（下采样到与HOG特征图相同的空间尺寸）
                gray_norm = gray.astype(np.float32) / 255.0
                gray_down = cv2.resize(gray_norm, (gray_feat_w, gray_feat_h))
                gray_feat = gray_down.flatten()
                # L2归一化
                gray_feat /= (np.linalg.norm(gray_feat) + 1e-6)

            # 3. 拼接特征
            combined_feat = np.concatenate([hog_feat.flatten(), gray_feat])

            if i == 0:
                out = np.zeros((combined_feat.size, n_scales), dtype=np.float32)

            out[:, i] = combined_feat * self.scale_window[i]

        return out

    def _detect_scale(self, frame, pos):
        """检测最佳尺度"""
        current_target_sz = self.base_target_sz * self.current_scale_factor
        scale_factors = self.scale_factors

        # 提取尺度样本（基于当前目标尺寸）
        xs = self._extract_scale_sample(frame, pos, current_target_sz, scale_factors)
        xsf = np.fft.fft(xs, axis=1)

        scale_response = np.real(np.fft.ifft(
            np.sum(self.sf_num * xsf, axis=0) / (self.sf_den + self.scale_lambda)
        ))

        recovered_scale = np.argmax(scale_response)

        self.current_scale_factor *= scale_factors[recovered_scale]

        self.current_scale_factor = np.clip(
            self.current_scale_factor,
            self.min_scale_factor,
            self.max_scale_factor
        )

    def _update_scale_model(self, frame, pos):
        """更新尺度模型"""
        current_target_sz = self.base_target_sz * self.current_scale_factor
        scale_factors = self.scale_factors

        # 提取尺度样本（基于当前目标尺寸）
        xs = self._extract_scale_sample(frame, pos, current_target_sz, scale_factors)
        xsf = np.fft.fft(xs, axis=1)

        new_sf_num = self.ysf * np.conj(xsf)
        new_sf_den = np.sum(xsf * np.conj(xsf), axis=0)

        if self.sf_num is None:
            self.sf_num = new_sf_num
            self.sf_den = new_sf_den
        else:
            self.sf_num = (1 - self.learning_rate_scale) * self.sf_num + self.learning_rate_scale * new_sf_num
            self.sf_den = (1 - self.learning_rate_scale) * self.sf_den + self.learning_rate_scale * new_sf_den


if __name__ == '__main__':
    ax_flag = False

    cap = cv2.VideoCapture("./video/output2.mp4")
    ret, frame = cap.read()

    if not ret:
        print("Failed to read video")
        exit()

    bbox = cv2.selectROI("AutoTrack4 Init", frame, False, False)
    cv2.destroyWindow("AutoTrack4 Init")

    tracker = AutoTrack4()
    tracker.init(frame, bbox)

    frame_idx = 0
    scale_history = []
    psr_history = []
    weight_history = []

    if ax_flag:
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        ax1.set_title("PSR over Time")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("PSR")
        ax1.axhline(y=tracker.psr_smooth_threshold, color='g', linestyle='--', label='PSR Threshold')
        ax1.legend()
        psr_x, psr_y = [], []
        line_psr, = ax1.plot([], [], 'b-', lw=2)

        ax2.set_title("Scale Factor")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Scale")
        scale_x, scale_y = [], []
        line_scale, = ax2.plot([], [], 'r-', lw=2)

        ax3.set_title("Kalman Fusion Weight")
        ax3.set_xlabel("Frame")
        ax3.set_ylabel("Weight")
        ax3.set_ylim(0, 1)
        weight_x, weight_y = [], []
        line_weight, = ax3.plot([], [], 'g-', lw=2)

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
        scale_history.append(tracker.current_scale_factor)
        psr_history.append(tracker.psr)
        weight_history.append(tracker._compute_fusion_weight(tracker.psr))

        if ax_flag:
            psr_x.append(frame_idx)
            psr_y.append(tracker.psr)
            line_psr.set_data(psr_x, psr_y)
            ax1.relim()
            ax1.autoscale_view()

            scale_x.append(frame_idx)
            scale_y.append(tracker.current_scale_factor)
            line_scale.set_data(scale_x, scale_y)
            ax2.relim()
            ax2.autoscale_view()

            weight_x.append(frame_idx)
            weight_y.append(tracker._compute_fusion_weight(tracker.psr))
            line_weight.set_data(weight_x, weight_y)
            ax3.relim()
            ax3.autoscale_view()

            heatmap.set_data(response)
            heatmap.set_clim(vmin=response.min(), vmax=response.max())

            plt.pause(0.001)

        # 绘制跟踪框
        psr = tracker.psr
        if psr > tracker.psr_smooth_threshold:
            color = (0, 255, 0)
        else:
            ratio = psr / tracker.psr_smooth_threshold
            color = (0, int(255 * ratio), int(255 * (1 - ratio)))

        cv2.rectangle(
            frame,
            (cx - w // 2, cy - h // 2),
            (cx + w // 2, cy + h // 2),
            color, 2
        )

        cv2.putText(frame, f"Frame: {frame_idx} | PSR: {psr:.2f} | Scale: {tracker.current_scale_factor:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("AutoTrack4: Scale + Kalman", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if ax_flag:
        plt.ioff()
        plt.show()