import numpy as np
import cv2


class STRCF:
    def __init__(self):
        # ========== 超参数（直接作为类属性） ==========
        self.padding = 2.0

        self.lambda_reg = 1e-2          # 空间正则 λ
        self.mu = 1.0                   # 时间正则 μ

        self.gamma_init = 1.0           # ADMM penalty
        self.gamma_max = 100.0
        self.gamma_step = 10.0
        self.admm_iter = 4

        self.output_sigma_factor = 0.1


        # ========== 运行时变量 ==========
        self.pos = None                 # [y, x]
        self.target_sz = None           # [h, w]
        self.window_sz = None           # [h, w]

        self.yf = None                  # 频域标签
        self.reg_window = None          # 空间正则权重

        self.cf_f = None                # 当前滤波器
        self.f_prev = None              # 上一帧滤波器
        self.g = None
        self.h = None

        self.cos_window = None


    # ==================================================
    # 初始化（第一帧）
    # ==================================================
    def init(self, frame, bbox):
        """
        bbox: [x, y, w, h]
        """
        x, y, w, h = bbox
        # print("x: ",x,", y: ", y,", w: ", w, ", h:", h, ", cx: ", x + w / 2, ", cy: ", y + h / 2)
        self.pos = np.array([y + h / 2, x + w / 2])
        self.target_sz = np.array([h, w])

        self.window_sz = np.floor(self.target_sz * (1 + self.padding)).astype(int)

        # cosine window
        self.cos_window = np.outer(
            np.hanning(self.window_sz[0]),
            np.hanning(self.window_sz[1])
        )

        # label
        output_sigma = np.sqrt(np.prod(self.target_sz)) * self.output_sigma_factor
        y = self._gaussian_label(self.window_sz, output_sigma)
        self.yf = np.fft.fft2(y)

        # spatial regularization
        self.reg_window = self._spatial_reg(self.window_sz)

        # initial features
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))

        self.cf_f = np.zeros_like(xf)
        self.f_prev = np.zeros_like(xf)
        self.g = np.zeros_like(xf)
        self.h = np.zeros_like(xf)

        self._train(frame)

    # ==================================================
    # 主跟踪函数
    # ==================================================
    def track(self, frame):
        # ---------- detection ----------
        disp = self._detect(frame)
        self.pos += disp

        #print("disp: ", disp)

        # ---------- model update ----------
        self._train(frame)

        return self.pos.copy(), self.target_sz.copy()

    # ==================================================
    # 检测
    # ==================================================
    def _detect(self, frame):
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))

        response_f = np.sum(np.conj(self.cf_f) * xf, axis=2)
        response = np.real(np.fft.ifft2(response_f))


        dy, dx = np.unravel_index(np.argmax(response), response.shape)
        dy -= response.shape[0] // 2
        dx -= response.shape[1] // 2



        return np.array([dy, dx])

    # ==================================================
    # 训练（严格对应 MATLAB 的 ADMM）
    # ==================================================
    def _train(self, frame):
        xf = self._get_features(frame, self.pos)
        xf = np.fft.fft2(xf, axes=(0, 1))
        print(xf.shape)
        yf = self.yf
        f_prev = self.f_prev

        gamma = self.gamma_init
        mu = self.mu

        T = xf.shape[0] * xf.shape[1]

        S_xx = np.sum(np.conj(xf) * xf, axis=2)
        Sfx_prev = np.sum(np.conj(xf) * f_prev, axis=2)

        f = self.cf_f
        g = self.g
        h = self.h

        for _ in range(self.admm_iter):
            # -------- f 子问题 --------
            B = S_xx + T * (gamma + mu)

            Sgx = np.sum(np.conj(xf) * g, axis=2)
            Shx = np.sum(np.conj(xf) * h, axis=2)

            term1 = (yf[..., None] * xf) / (T * (gamma + mu))
            term2 = -h / (gamma + mu)
            term3 = (gamma * g) / (gamma + mu)
            term4 = (mu * f_prev) / (gamma + mu)

            num = (
                (xf * (S_xx * yf)[..., None]) / (T * (gamma + mu))
                + (mu * xf * Sfx_prev[..., None]) / (gamma + mu)
                - (xf * Shx[..., None]) / (gamma + mu)
                + (gamma * xf * Sgx[..., None]) / (gamma + mu)
            )

            f = term1 + term2 + term3 + term4 - num / B[..., None]

            # -------- g 子问题（空间正则）--------
            f_spatial = np.real(np.fft.ifft2(gamma * f + h, axes=(0, 1)))
            g = np.fft.fft2(
                f_spatial / (gamma + self.lambda_reg * self.reg_window[..., None] ** 2),
                axes=(0, 1)
            )

            # -------- h 更新 --------
            h = h + gamma * (f - g)

            gamma = min(gamma * self.gamma_step, self.gamma_max)

        self.cf_f = f
        self.f_prev = f
        self.g = g
        self.h = h

    # ==================================================
    # 特征（灰度，单通道）
    # ==================================================
    def _get_features(self, frame, pos):
        y, x = pos.astype(int)
        h, w = self.window_sz

        # print("cx: ", x, ", cy: ", y)

        y1 = y - h // 2
        y2 = y1 + h
        x1 = x - w // 2
        x2 = x1 + w

        # clamp
        y1c, y2c = max(0, y1), min(frame.shape[0], y2)
        x1c, x2c = max(0, x1), min(frame.shape[1], x2)


        patch = frame[y1c:y2c, x1c:x2c]

        # ---- padding to exact size ----
        pad_top = y1c - y1
        pad_bottom = y2 - y2c
        pad_left = x1c - x1
        pad_right = x2 - x2c

        if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
            patch = cv2.copyMakeBorder(
                patch,
                pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REPLICATE
            )

        # now patch.shape[:2] == window_sz

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray -= gray.mean()

        return gray[..., None] * self.cos_window[..., None]

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


if __name__ == '__main__':
    cap = cv2.VideoCapture("./video/1.mp4")  # 改成视频路径也可以
    #cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    print("frame size: ", frame.shape)
    bbox = cv2.selectROI("SRDCF Init", frame, False, False)
    cv2.destroyWindow("SRDCF Init")

    tracker = STRCF()
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        pos, target_sz = tracker.track(frame)

        cy, cx = pos.astype(int)  # pos = [y, x]
        h, w = target_sz.astype(int)  # size = [h, w]

        #print("cx: ", cx, " cy: ", cy, " w: ", w, " h: ", h)

        cv2.rectangle(
            frame,
            (cx - w // 2, cy - h // 2),
            (cx + w // 2, cy + h // 2),
            (0, 255, 0),
            2
        )



        cv2.imshow("SRDCF Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()