import numpy as np
import cv2


class STRCF:
    def __init__(self):
        # ========== 超参数 ==========
        self.padding = 1.0

        self.lambda_reg = 1e-2

        self.mu = 2.0

        self.gamma_init = 1.0
        self.gamma_max = 10.0
        self.gamma_step = 10.0
        self.admm_iter = 4

        self.output_sigma_factor = 0.1


        # ===== HOG 参数 =====
        self.hog_cell_size = 4
        self.hog_nbins = 9
        self.hog = None

        # ========== 状态 ==========
        self.pos = None
        self.target_sz = None
        self.window_sz = None

        self.yf = None
        self.reg_window = None

        self.cf_f = None
        self.f_prev = None
        self.g = None
        self.h = None

        self.cos_window = None


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

        # -------- HOG 尺寸（cell grid）--------
        #Hc和Wc表示HOG特征的宽度和长度

        cs = self.hog_cell_size
        Hc = self.window_sz[0] // cs
        Wc = self.window_sz[1] // cs
        self.Hc = Hc
        self.Wc = Wc

        print("Hc:", self.Hc, "Wc:", self.Wc)

        self.window_sz = [Hc * cs, Wc * cs]
        # print("window_sz:", self.window_sz)
        # print("target_sz:", self.target_sz)


        # -------- cosine window（cell 级）--------
        self.cos_window = np.outer(np.hanning(Hc), np.hanning(Wc))

        # -------- label（cell 级）--------
        output_sigma = np.sqrt(np.prod(self.target_sz)) * self.output_sigma_factor / cs
        y = self._gaussian_label((Hc, Wc), output_sigma)
        self.yf = np.fft.fft2(y)

        # -------- spatial regularization（cell 级）--------
        self.reg_window = self._spatial_reg((Hc, Wc))

        # -------- 初始化 HOG（只做一次）--------
        self.hog = cv2.HOGDescriptor(
            _winSize=(self.window_sz[1], self.window_sz[0]),
            _blockSize=(cs, cs),
            _blockStride=(cs, cs),
            _cellSize=(cs, cs),
            _nbins=self.hog_nbins
        )

        # -------- 初始化滤波器 --------
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))

        self.cf_f = np.zeros_like(xf)
        self.f_prev = np.zeros_like(xf)
        self.g = np.zeros_like(xf)
        self.h = np.zeros_like(xf)

        for i in range(5):
            self._train(frame)


    # ==================================================
    # 主跟踪
    # ==================================================
    def track(self, frame):
        disp = self._detect(frame)
        #print("disp:", disp)
        self.pos += disp
        self._train(frame)
        return self.pos.copy(), self.target_sz.copy()


    # ==================================================
    # 检测
    # ==================================================
    def _detect(self, frame):
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))
        # print("xf shape:", xf.shape)
        # print("cf_f shape:", self.cf_f.shape)

        response_f = np.sum(np.conj(self.cf_f) * xf, axis=2)
        response = np.real(np.fft.ifft2(response_f))
        # print("response shape:", response.shape)

        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        dy -= response.shape[0] // 2
        dx -= response.shape[1] // 2

        disp = np.array([dy, dx], dtype=np.float32)
        disp *= self.hog_cell_size

        return disp


    # ==================================================
    # 训练（STRCF ADMM）
    # ==================================================
    def _train(self, frame):
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))
        #print(xf.shape)

        yf = self.yf
        f_prev = self.f_prev

        gamma = self.gamma_init
        mu = self.mu

        T = xf.shape[0] * xf.shape[1]

        S_xx = np.sum(np.conj(xf) * xf, axis=2)
        Sfx_prev = np.sum(np.conj(xf) * f_prev, axis=2)

        f, g, h = self.cf_f, self.g, self.h

        for _ in range(self.admm_iter):
            B = S_xx + T * (gamma + mu)

            Sgx = np.sum(np.conj(xf) * g, axis=2)
            Shx = np.sum(np.conj(xf) * h, axis=2)

            f = (
                (yf[..., None] * xf) / (T * (gamma + mu))
                - h / (gamma + mu)
                + (gamma * g) / (gamma + mu)
                + (mu * f_prev) / (gamma + mu)
                - (
                    (xf * (S_xx * yf)[..., None]) / (T * (gamma + mu))
                    + (mu * xf * Sfx_prev[..., None]) / (gamma + mu)
                    - (xf * Shx[..., None]) / (gamma + mu)
                    + (gamma * xf * Sgx[..., None]) / (gamma + mu)
                ) / B[..., None]
            )

            f_spatial = np.real(np.fft.ifft2(gamma * f + h, axes=(0, 1)))
            g = np.fft.fft2(
                f_spatial / (gamma + self.lambda_reg * self.reg_window[..., None] ** 2),
                axes=(0, 1)
            )

            h = h + gamma * (f - g)
            gamma = min(gamma * self.gamma_step, self.gamma_max)


        self.cf_f = f
        self.f_prev = f
        self.g = g
        self.h = h


    # ==================================================
    # HOG 特征（cell grid）
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
        hog_feat = hog_feat.reshape(
            self.Wc,
            self.Hc,
            self.hog_nbins
        ).astype(np.float32).transpose((1, 0, 2))
        # hog_feat = hog_feat.reshape(
        #         self.Hc,
        #         self.Wc,
        #         self.hog_nbins
        #     ).astype(np.float32)

        hog_feat -= hog_feat.mean(axis=(0, 1), keepdims=True)
        hog_feat *= self.cos_window[..., None]

        return hog_feat

    # ==================================================
    # 工具函数
    # ==================================================
    def _gaussian_label(self, sz, sigma):
        h, w = sz
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        return np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2) / sigma ** 2)

    # def _spatial_reg(self, sz):
    #     h, w = sz
    #     y, x = np.ogrid[:h, :w]
    #     cy, cx = h / 2, w / 2
    #     return ((y - cy) ** 2 + (x - cx) ** 2) / max(h, w) ** 2

    def _spatial_reg(self, sz):
        h, w = sz
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        dist = (y - cy) ** 2 + (x - cx) ** 2
        return np.exp(dist / (0.5 * h * w))


# ==================================================
# Demo
# ==================================================
if __name__ == '__main__':
    cap = cv2.VideoCapture("./video/1.mp4")
    #cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    bbox = cv2.selectROI("STRCF Init", frame, False, False)
    cv2.destroyWindow("STRCF Init")

    tracker = STRCF()
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

        cv2.imshow("STRCF HOG Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
