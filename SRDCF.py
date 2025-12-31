import numpy as np
import cv2

class HOG:
    def __init__(self, winsize):
        self.winsize = winsize
        self.block_size = (8, 8)
        self.block_stride = (4, 4)
        self.cell_size = (4, 4)
        self.nbins = 9

        self.hog = cv2.HOGDescriptor(
            self.winsize,
            self.block_size,
            self.block_stride,
            self.cell_size,
            self.nbins
        )

    def get_feature(self, image):
        hist = self.hog.compute(image, self.winsize, padding=(0, 0))
        w, h = self.winsize
        sw, sh = self.block_stride
        w = w // sw - 1
        h = h // sh - 1
        return hist.reshape(w, h, 36).transpose(2, 1, 0)


def gaussian_label(sz, sigma):
    h, w = sz
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    g = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))
    return np.roll(np.roll(g, -cy, axis=0), -cx, axis=1)


def spatial_weight(sz, alpha=2.0):
    h, w = sz
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist = ((y - cy)**2 + (x - cx)**2) / (h * w)
    dist = ((y - cy)/h)**2 + ((x - cx)/w)**2
    return 1.0 + alpha * dist


class SRDCF:
    def __init__(self,lr=0.025,mu=2.0, admm_iters=10):
        self.max_patch_size = 256
        self.lr = lr
        self.mu = mu
        self.admm_iters = admm_iters


    def init(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        cx = x + w // 2
        cy = y + h // 2
        roi = (cx, cy, w, h)  # 转换为中心点+长宽的形式

        self.scale = self.max_patch_size / float(max(h, w))
        self.scale = min(self.scale, 2.0)
        self.pw = int(w * self.scale) // 4 * 4 + 4
        self.ph = int(h * self.scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        patch = frame[y:y+h, x:x+w]
        patch = cv2.resize(patch, (self.pw, self.ph))
        print("w: ", w, " h: ", h)
        print("pw: ", self.pw, " ph: ", self.ph)
        feat = self.hog.get_feature(patch).astype(np.float32)
        print("patch size:", feat.shape)
        self.x_hat = np.fft.fft2(feat, axes=(-2, -1))

        D, H, W = feat.shape
        self.sz = (H, W)
        print("sz: ", self.sz)
        self.w_spatial = spatial_weight(self.sz)

        sigma = 0.1 * min(self.sz)
        self.y = gaussian_label(self.sz, sigma=sigma)
        self.y_hat = np.fft.fft2(self.y)

        self.w_spatial = spatial_weight(self.sz)

        self.f = np.zeros((D, H, W), np.float32)
        self.g_hat = np.zeros((D, H, W), np.complex64)
        self.zeta_hat = np.zeros((D, H, W), np.complex64)

        self._admm_optimize()
        self.roi = roi

    def _admm_optimize(self):
        for _ in range(self.admm_iters):
            self._update_g()
            self._update_f()
            self._update_zeta()

    def _update_g(self):
        f_hat = np.fft.fft2(self.f, axes=(-2, -1))
        x = self.x_hat

        xx = np.sum(np.conj(x) * x, axis=0)
        xf = np.sum(np.conj(x) * f_hat, axis=0)
        xz = np.sum(np.conj(x) * self.zeta_hat, axis=0)
        xy = np.conj(x) * self.y_hat[None, : , :]

        #term1 = (self.y_hat * x - self.zeta_hat + self.mu * f_hat) / self.mu
        term1 = (xy - self.zeta_hat + self.mu * f_hat) / self.mu
        denom = self.mu + xx

        # term2 = (x / (self.mu * denom)) * (
        #     self.y_hat * xx - xz + self.mu * xf
        # )

        term2 = (x / (self.mu * denom)) * (
                np.conj(self.y_hat * xx) - xz + self.mu * xf
        )

        self.g_hat = term1 - term2

    def _update_f(self):
        rhs = np.fft.ifft2(
            self.g_hat + self.zeta_hat / self.mu,
            axes=(-2, -1)
        )
        rhs = np.real(rhs)
        self.f = (self.mu * rhs) / (self.w_spatial**2 + self.mu)

    def _update_zeta(self):
        self.zeta_hat += self.mu * (
            self.g_hat - np.fft.fft2(self.f, axes=(-2, -1))
        )

    def detect(self, frame):

        cx, cy, w, h = self.roi

        x1 = int(cx - w//2)
        y1 = int(cy - h//2)
        x1 = max(0, x1)
        y1 = max(0, y1)

        patch = frame[y1:y1+h, x1:x1+w]
        patch = cv2.resize(patch, self.hog.winsize)

        feat = self.hog.get_feature(patch).astype(np.float32)
        x_hat = np.fft.fft2(feat, axes=(-2, -1))

        # resp = np.sum(
        #     np.fft.ifft2(
        #         x_hat * np.fft.fft2(self.f, axes=(-2, -1)),
        #         axes=(-2, -1)
        #     ),
        #     axis=0
        # )
        resp = np.sum(
            np.fft.ifft2(
                x_hat * np.conj(np.fft.fft2(self.f, axes=(-2, -1))),
                axes=(-2, -1)
            ),
            axis=0
        )

        resp = np.real(resp)
        dy, dx = np.unravel_index(np.argmax(resp), resp.shape)
        dy -= resp.shape[0] // 2
        dx -= resp.shape[1] // 2

        sx, sy = self.hog.block_stride
        dx_pix = int(dx * sx)
        dy_pix = int(dy * sy)

        self.roi = (cx + dx_pix, cy + dy_pix, w, h)

        return self.roi

    def update(self, frame):

        cx, cy, w, h = self.roi
        x1 = int(cx - w//2)
        y1 = int(cy - h//2)
        x1 = max(0, x1)
        y1 = max(0, y1)

        patch = frame[y1:y1+h, x1:x1+w]
        patch = cv2.resize(patch, self.hog.winsize)

        feat = self.hog.get_feature(patch).astype(np.float32)
        self.x_hat = np.fft.fft2(feat, axes=(-2, -1))

        f_old = self.f.copy()
        self._admm_optimize()
        self.f = (1 - self.lr) * f_old + self.lr * self.f




def main():
    cap = cv2.VideoCapture("./video/1.mp4")  # 改成视频路径也可以
    #cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return

    bbox = cv2.selectROI("SRDCF Init", frame, False, False)
    cv2.destroyWindow("SRDCF Init")

    tracker = SRDCF()
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, w, h = tracker.detect(frame)
        tracker.update(frame)


        cv2.rectangle(
            frame,
            (cx - w//2, cy - h//2),
            (cx + w//2, cy + h//2),
            (0, 255, 0),
            2
        )

        cv2.imshow("SRDCF Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
