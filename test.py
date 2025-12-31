import numpy as np
import cv2

class HOGFeature:
    def __init__(
        self,
        win_size,
        cell_size=(4, 4),
        block_size=(8, 8),
        block_stride=(4, 4),
        nbins=9,
        eps=1e-6
    ):
        self.win_w, self.win_h = win_size
        self.cell_w, self.cell_h = cell_size
        self.block_w, self.block_h = block_size
        self.stride_w, self.stride_h = block_stride
        self.nbins = nbins
        self.eps = eps

        # cell / block 维度
        self.n_cells_x = self.win_w // self.cell_w
        self.n_cells_y = self.win_h // self.cell_h

        self.cells_per_block_x = self.block_w // self.cell_w
        self.cells_per_block_y = self.block_h // self.cell_h

        self.angle_unit = 180 / self.nbins

    def compute(self, gray):
        """
        gray: (H, W), uint8 or float32
        return: HOG feature vector (1D)
        """
        if gray.ndim != 2:
            raise ValueError("Input must be grayscale image")

        # 1. 梯度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)

        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180  # unsigned

        # 2. 计算 cell 直方图
        cell_hist = np.zeros(
            (self.n_cells_y, self.n_cells_x, self.nbins),
            dtype=np.float32
        )

        for cy in range(self.n_cells_y):
            for cx in range(self.n_cells_x):
                y0 = cy * self.cell_h
                y1 = y0 + self.cell_h
                x0 = cx * self.cell_w
                x1 = x0 + self.cell_w

                mag_patch = magnitude[y0:y1, x0:x1]
                ang_patch = angle[y0:y1, x0:x1]

                for i in range(self.cell_h):
                    for j in range(self.cell_w):
                        bin_idx = int(ang_patch[i, j] // self.angle_unit) % self.nbins
                        cell_hist[cy, cx, bin_idx] += mag_patch[i, j]

        # 3. block 归一化（L2-Hys）
        hog_feats = []

        for y in range(0, self.n_cells_y - self.cells_per_block_y + 1,
                       self.stride_h // self.cell_h):
            for x in range(0, self.n_cells_x - self.cells_per_block_x + 1,
                           self.stride_w // self.cell_w):

                block = cell_hist[
                    y:y + self.cells_per_block_y,
                    x:x + self.cells_per_block_x,
                    :
                ].reshape(-1)

                # L2-Hys
                norm = np.sqrt(np.sum(block ** 2) + self.eps ** 2)
                block = block / norm
                block = np.clip(block, 0, 0.2)
                block = block / (np.sqrt(np.sum(block ** 2)) + self.eps)

                hog_feats.append(block)

        return np.concatenate(hog_feats)


if __name__ == '__main__':
    # 参数

    img = cv2.imread("picture/pic1.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    w, h = img.shape
    win_size = (w, h)

    # OpenCV HOG
    hog_cv = cv2.HOGDescriptor(win_size,(8, 8),(4, 4),(4, 4),9)
    feat_cv = hog_cv.compute(img)

    # Python HOG
    hog_py = HOGFeature(win_size)
    feat_py = hog_py.compute(img)

    print("OpenCV HOG dim:", feat_cv.shape)
    print("Python HOG dim:", feat_py.shape)

    # 数值对比
    diff = np.linalg.norm(feat_cv - feat_py) / np.linalg.norm(feat_cv)
    print("Relative L2 error:", diff)
