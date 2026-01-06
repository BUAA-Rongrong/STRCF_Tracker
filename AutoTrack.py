import numpy as np
import cv2


class AutoTrack:
    def __init__(self):
        # ================= 超参数 =================
        self.padding = 1.0
        self.lambda_reg = 1e-2

        # 时间正则（AutoTrack）
        self.ref_mu = 13.0
        self.epsilon = 1.0
        self.mu = 0
        self.delta = 0.2 #响应变化调整系数
        self.phi = 0.3 #更新阈值

        #更新ref_mu
        self.zeta = 13.0
        self.nu = 0.2

        # ADMM
        self.gamma_init = 1.0
        self.gamma_max = 100.0
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

        for _ in range(5):
            self._train(frame)

    # ==================================================
    # 跟踪
    # ==================================================
    def track(self, frame):
        #1.检测阶段，得到响应图
        disp, response = self._detect(frame)
        print("disp: ", disp)
        self.pos += disp


        self._train(frame, response, disp)
        return self.pos.copy(), self.target_sz.copy()

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

        gamma = self.gamma_init
        mu = self.mu
        occ = False

        if response is not None and self.response_prev is not None:
            response_diff = self._align_and_diff_response(response, disp)
            ref_mu, occ = self._update_ref_mu(response_diff)
            self.ref_mu = ref_mu

            print("occ: ", occ)


            delta_resp = self.delta * np.log(1 + response_diff)
            delta_resp = cv2.blur(delta_resp, (3, 3))
            self.reg_window1 = np.clip(self.reg_window + delta_resp, self.reg_min, self.reg_max)


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
            g_l_spatial = np.real(np.fft.ifft2(self.g_f + self.l_f, axes=(0, 1)))
            denom = self.lambda_reg * self.reg_window1[..., None] ** 2 + gamma * T
            self.h_f = np.fft.fft2((T / denom) * g_l_spatial, axes=(0, 1))

            # ===== mu 更新（时间正则）=====
            diff = np.sum(np.abs(self.g_f - self.g_pre) ** 2)
            z = diff / (2 * self.epsilon)
            mu = self.ref_mu - z
            #mu = max(self.ref_mu - z, 0)

            # ===== 拉格朗日乘子 =====
            self.l_f += gamma * (self.g_f - self.h_f)
            gamma = min(gamma * self.gamma_step, self.gamma_max)

        self.g_pre = self.g_f.copy()
        self.mu = mu

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
        dy, dx = disp[0], disp[1]
        dy_p, dx_p = self.disp_prev[0], self.disp_prev[1]

        r1 = np.roll(response, (-dy, -dx), axis=(0, 1))
        r2 = np.roll(self.response_prev, (-dy_p, -dx_p), axis=(0, 1))

        return np.abs(np.abs(r1) - np.abs(r2)) / (np.abs(r2) + 1e-6)


    def _update_ref_mu(self, response_diff):

        m = self.zeta
        p = self.nu
        eta = np.linalg.norm(response_diff.ravel(), 2) / 1e4

        if eta < self.phi:
            return m / (1 + np.log(p * eta + 1)), False
        else:
            return 50.0, True




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
    cap = cv2.VideoCapture("./video/1.mp4")
    #cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    bbox = cv2.selectROI("AutoTrack Init", frame, False, False)
    cv2.destroyWindow("AutoTrack Init")

    tracker = AutoTrack()
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos, target_sz = tracker.track(frame)
        cy, cx = pos.astype(int)
        h, w = target_sz.astype(int)

        cv2.rectangle(
            frame,
            (cx - w // 2, cy - h // 2),
            (cx + w // 2, cy + h // 2),
            (0, 255, 0), 2
        )

        cv2.imshow("AutoTrack HOG Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
