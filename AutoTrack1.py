import numpy as np
import cv2
import matplotlib.pyplot as plt

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

        # AutoTrack variables
        self.g_f = None      # 主滤波器
        self.h_f = None      # 空间正则变量
        self.l_f = None      # 拉格朗日乘子
        self.g_pre = None

        self.cos_window = None
        self.response_prev = None
        self.disp_prev = None

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
        #self.reg_window = self._init_reg_window1((Hc, Wc), 0.5, 1, 100)
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

    # ==================================================
    # 跟踪
    # ==================================================
    def track(self, frame):
        #1.检测阶段，得到响应图
        disp, response = self._detect(frame)
        #print("disp: ", disp)
        self.pos += disp

        #2.从第2帧开始，更新ref_mu
        occ = False

        psr = self.compute_psr(response, 5)
        #print("psr: ", psr)
        # print("response: ", response)
        # print("response_pre: ", self.response_prev)
        if response is not None and self.response_prev is not None:
            response_diff = self._align_and_diff_response(response, disp)
            ref_mu, occ = self._update_ref_mu(response_diff)
            self.ref_mu = ref_mu
            self.mu = self.zeta
            print("occ: ", occ)
            delta_resp = self.delta * np.log(1 + response_diff)
            delta_resp = cv2.blur(delta_resp, (3, 3))
            self.reg_window1 = np.clip(self.reg_window + delta_resp, self.reg_min, self.reg_max)

        #3.当响应值小于阈值则进行训练
        if occ == False:
            self._train(frame, response, disp)

        return self.pos.copy(), self.target_sz.copy(), response

    # ==================================================
    # 检测
    # ==================================================
    def _detect(self, frame):
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))#特征图进行傅里叶变换

        response_f = np.sum(np.conj(self.g_f) * xf, axis=2)
        response = np.real(np.fft.ifft2(response_f))#对响应图进行傅里叶逆变换

        dy, dx = np.unravel_index(np.argmax(response), response.shape)#找到响应图峰值
        dy -= response.shape[0] // 2
        dx -= response.shape[1] // 2

        disp = np.array([dy, dx], dtype=np.float32) * self.hog_cell_size#转换为实际位置偏移
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

        for _ in range(self.admm_iter):
            # ===== g 子问题 =====
            B = S_xx + T * (gamma + mu)

            Shx = np.sum(np.conj(xf) * self.h_f, axis=2)
            Slx = np.sum(np.conj(xf) * self.l_f, axis=2)

            term1 = (yf[..., None] * xf) / (T * (gamma + mu))
            term2 = - self.l_f / (gamma + mu)
            term3 = (gamma * self.h_f) / (gamma + mu)
            term4 = (mu * self.g_pre) / (gamma + mu)

            corr = (
                (xf * (S_xx * yf)[..., None]) / (T * (gamma + mu))
                + (mu * xf * Sgx_pre[..., None]) / (gamma + mu)
                - (xf * Slx[..., None]) / (gamma + mu)
                + (gamma * xf * Shx[..., None]) / (gamma + mu)
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
            #mu = max(mu, 1e-6)

            # print("diff: ", diff)
            # print("mu: ", mu, "ref mu: ", self.ref_mu)

            # ===== 拉格朗日乘子 =====
            self.l_f += gamma * (self.g_f - self.h_f)
            gamma = min(gamma * self.gamma_step, self.gamma_max)



        self.g_pre = self.g_f

        self.response_prev = response
        self.disp_prev = disp

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
        #print("eta: ", eta)
        self.eta = eta

        if eta < self.phi:
            return m / (1 + np.log(p * eta + 1)), False
        else:
            return 50.0, True

    def compute_psr(self, response, peak_size=5):
        """
        Compute Peak-to-Sidelobe Ratio (PSR)

        Parameters
        ----------
        response : ndarray (H, W)
        peak_size : int
            window size to exclude around the peak

        Returns
        -------
        psr : float
        """

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
        print("psr: ", psr)
        return psr

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

    def _init_reg_window1(self, window_sz, ratio, reg_window_min, reg_window_max):
        H, W = window_sz
        reg_window = np.ones((H, W), np.float32) * reg_window_max

        ch = int(np.round(H * ratio))
        cw = int(np.round(W * ratio))

        ch = max(1, min(ch, W))
        cw = max(1, min(cw, W))

        cy = (H - 1) // 2
        cx = (W - 1) // 2

        h_start = cy - ch // 2
        h_end = h_start + ch
        w_start = cx - cw // 2
        w_end = w_start + cw

        h_start = max(0, h_start)
        w_start = max(0, w_start)
        h_end = min(H, h_end)
        w_end = min(W, w_end)

        reg_window[h_start:h_end, w_start:w_end] = reg_window_min

        print(reg_window.shape)
        return reg_window


if __name__ == '__main__':

    ax_flag = True

    cap = cv2.VideoCapture("./video/4.mp4")
    #cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    bbox = cv2.selectROI("AutoTrack Init", frame, False, False)
    cv2.destroyWindow("AutoTrack Init")

    tracker = AutoTrack()
    tracker.init(frame, bbox)

    frame_idx = 0

    if ax_flag:
        # ===== eta 曲线 =====
        etas = []
        plt.ion()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # --- 左：eta ---
        ax1.set_title("Eta over Time")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Eta")
        eta_x, eta_y = [], []
        line_eta, = ax1.plot([], [], lw=2)

        # --- 右：response heatmap ---
        ax2.set_title("Response Heatmap")
        heatmap = ax2.imshow(
            np.zeros((tracker.Hc, tracker.Wc)),
            cmap="jet",
            interpolation="nearest",
            origin="upper"
        )
        plt.colorbar(heatmap, ax=ax2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos, target_sz, response = tracker.track(frame)
        cy, cx = pos.astype(int)
        h, w = target_sz.astype(int)

        frame_idx += 1

        if ax_flag:
            # ===== eta 更新 =====
            eta = tracker.eta
            eta_x.append(len(eta_x))
            eta_y.append(eta)
            line_eta.set_data(eta_x, eta_y)
            ax1.relim()
            ax1.autoscale_view()

            # ===== response heatmap 更新 =====
            heatmap.set_data(response)
            heatmap.set_clim(vmin=response.min(), vmax=response.max())

            plt.pause(0.001)

        cv2.rectangle(
            frame,
            (cx - w // 2, cy - h // 2),
            (cx + w // 2, cy + h // 2),
            (0, 255, 0), 2
        )

        cv2.putText(frame, f"Frame: {frame_idx}", (cx - w // 2, cy - h//2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("AutoTrack HOG Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # plt.plot(etas)
    # plt.show()

    cap.release()
    cv2.destroyAllWindows()
