import numpy as np
import cv2


class AutoTrack:
    def __init__(self):
        # ================= 超参数（论文推荐值）=================
        self.padding = 2.0  # 论文使用2.5倍搜索区域
        self.lambda_reg = 1e-2  # 空间正则化强度

        # AutoTrack 时空正则化参数（论文Section 4.1, 4.2）
        self.delta = 0.2  # 论文: δ = 0.2，局部响应变化权重
        self.nu = 2e-5  # 论文: ν = 2×10^-5，全局响应变化权重
        self.zeta = 13.0  # 论文: ζ = 13，时间正则化系数
        self.phi = 3000  # 论文: φ = 3000，响应变化阈值

        # 时间正则化初始值
        self.theta_t = 15.0  # 论文STRCF基线使用θ=15
        self.theta_ref = 15.0

        # ADMM优化参数（论文Section 4.2）
        self.gamma = 1.0
        self.gamma_max = 10000.0  # 论文: 10000
        self.gamma_step = 10.0  # 论文: β = 10
        self.admm_iter = 4  # 论文: 4次迭代

        # 高斯标签参数
        self.output_sigma_factor = 0.1

        # HOG特征参数
        self.hog_cell_size = 4
        self.hog_nbins = 9
        self.hog = None

        # ================= 跟踪状态 =================
        self.pos = None  # 目标中心位置 [y, x]
        self.target_sz = None  # 目标大小 [h, w]
        self.window_sz = None  # 搜索窗口大小

        # 标签和窗口
        self.yf = None  # 高斯标签（频域）
        self.cos_window = None  # 余弦窗

        # 空间正则化（论文公式3）
        self.u_spatial = None  # 基础碗形空间正则化
        self.u_tilde = None  # 自动空间正则化

        # 滤波器（论文公式5-15）
        self.g_f = None  # 主滤波器 G_t（频域）
        self.h_f = None  # 辅助变量 H_t
        self.m_f = None  # 拉格朗日乘子 M_t
        self.g_f_prev = None  # 前一帧滤波器 G_{t-1}

        # 响应图（论文公式2）
        self.response_prev = None
        self.max_pos_prev = None

        # 是否初始化完成
        self.is_initialized = False

    # ==================================================
    # 初始化（论文Algorithm流程）
    # ==================================================
    def init(self, frame, bbox):
        """
        初始化跟踪器
        bbox: (x, y, w, h) - OpenCV格式
        """
        x, y, w, h = bbox

        # 确保尺寸是cell_size的倍数
        w = (w // self.hog_cell_size) * self.hog_cell_size
        h = (h // self.hog_cell_size) * self.hog_cell_size

        # 保存目标状态（注意：pos是[y, x]格式）
        self.pos = np.array([y + h / 2, x + w / 2], dtype=np.float32)
        self.target_sz = np.array([h, w], dtype=np.float32)

        # 计算搜索窗口大小
        self.window_sz = np.floor(self.target_sz * (1 + self.padding)).astype(int)

        # 确保窗口大小是cell_size的倍数
        cs = self.hog_cell_size
        Hc = self.window_sz[0] // cs
        Wc = self.window_sz[1] // cs
        self.Hc, self.Wc = Hc, Wc
        self.window_sz = np.array([Hc * cs, Wc * cs])

        # 创建余弦窗（减少边界效应）
        self.cos_window = np.outer(np.hanning(Hc), np.hanning(Wc))

        # 创建高斯标签（论文公式1中的y）
        output_sigma = np.sqrt(np.prod(self.target_sz)) * self.output_sigma_factor / cs
        y = self._gaussian_label((Hc, Wc), output_sigma)
        self.yf = np.fft.fft2(y)

        # 初始化空间正则化（论文公式3，基于SRDCF的u）
        self.u_spatial = self._init_spatial_regularization((Hc, Wc))
        self.u_tilde = self.u_spatial.copy()

        # 初始化HOG描述符
        self.hog = cv2.HOGDescriptor(
            (self.window_sz[1], self.window_sz[0]),
            (cs, cs), (cs, cs), (cs, cs),
            self.hog_nbins
        )

        # 提取第一帧特征
        x_spatial = self._get_features(frame, self.pos)
        xf = np.fft.fft2(x_spatial, axes=(0, 1))

        # 初始化ADMM变量
        self.g_f = np.zeros_like(xf, dtype=np.complex128)
        self.h_f = np.zeros_like(xf, dtype=np.complex128)
        self.m_f = np.zeros_like(xf, dtype=np.complex128)
        self.g_f_prev = np.zeros_like(xf, dtype=np.complex128)

        # 初始化训练（论文建议多次迭代以稳定滤波器）
        for _ in range(5):
            self._train_admm(xf, is_first_frame=True)

        # 计算初始响应图
        response = self._compute_response(xf, self.g_f)
        self.response_prev = response
        self.max_pos_prev = np.unravel_index(np.argmax(response), response.shape)

        self.is_initialized = True
        print(f"[AutoTrack] 初始化完成: pos={self.pos}, target_sz={self.target_sz}")

    # ==================================================
    # 跟踪更新（论文主流程）
    # ==================================================
    def track(self, frame):
        """
        跟踪当前帧
        返回: (cy, cx), (h, w) - 中心位置和目标大小
        """
        if not self.is_initialized:
            raise RuntimeError("跟踪器未初始化，请先调用init()")

        # 1. 检测阶段（论文公式16）
        x_spatial = self._get_features(frame, self.pos)
        xf = np.fft.fft2(x_spatial, axes=(0, 1))
        response = self._compute_response(xf, self.g_f)

        # 2. 定位目标（找到响应图最大值）
        max_pos = np.unravel_index(np.argmax(response), response.shape)
        dy = (max_pos[0] - response.shape[0] // 2) * self.hog_cell_size
        dx = (max_pos[1] - response.shape[1] // 2) * self.hog_cell_size

        # 更新目标位置
        self.pos[0] += dy
        self.pos[1] += dx

        # 3. 计算响应变化（论文公式2）
        Pi_local, Pi_global = self._compute_response_variation(
            response, self.response_prev, max_pos, self.max_pos_prev
        )

        print(f"[AutoTrack] 位移=({dy:.1f}, {dx:.1f}), "
              f"全局变化={Pi_global:.2f}, 阈值={self.phi}")

        # 4. 自动时空正则化（论文Section 4.1）
        should_update = True

        if Pi_global > self.phi:
            # 响应变化过大，检测到异常，停止学习（论文公式4）
            print(f"[AutoTrack] 检测到异常（变化={Pi_global:.2f} > {self.phi}），停止更新")
            should_update = False
        else:
            # 4.1 自动空间正则化（论文公式3）
            self.u_tilde = self._compute_adaptive_spatial_regularization(Pi_local)

            # 4.2 自动时间正则化参考值（论文公式4）
            self.theta_ref = self._compute_temporal_regularization_ref(Pi_global)

        # 5. 更新滤波器（论文Section 4.2，公式5-15）
        if should_update:
            # 重新提取当前位置的特征
            x_spatial = self._get_features(frame, self.pos)
            xf = np.fft.fft2(x_spatial, axes=(0, 1))

            # ADMM优化训练
            self._train_admm(xf, is_first_frame=False)

        # 6. 更新历史响应
        self.response_prev = response
        self.max_pos_prev = max_pos

        return self.pos.copy(), self.target_sz.copy()

    # ==================================================
    # 特征提取
    # ==================================================
    def _get_features(self, frame, pos):
        """
        提取HOG特征
        """
        y, x = pos.astype(int)
        h, w = self.window_sz

        # 计算提取区域
        y1, y2 = y - h // 2, y - h // 2 + h
        x1, x2 = x - w // 2, x - w // 2 + w

        # 边界裁剪
        y1c, y2c = max(0, y1), min(frame.shape[0], y2)
        x1c, x2c = max(0, x1), min(frame.shape[1], x2)

        # 提取patch
        patch = frame[y1c:y2c, x1c:x2c]

        # 边界填充
        patch = cv2.copyMakeBorder(
            patch,
            y1c - y1, y2 - y2c,
            x1c - x1, x2 - x2c,
            cv2.BORDER_REPLICATE
        )

        # 转换为灰度图
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch

        # 提取HOG特征
        hog_feat = self.hog.compute(gray)

        # 重塑为3D张量 [H, W, C]
        hog_feat = hog_feat.reshape(self.Wc, self.Hc, self.hog_nbins)
        hog_feat = hog_feat.transpose(1, 0, 2).astype(np.float32)

        # 归一化和加窗
        hog_feat -= hog_feat.mean(axis=(0, 1), keepdims=True)
        hog_feat *= self.cos_window[..., None]

        return hog_feat

    # ==================================================
    # 响应计算（论文公式16）
    # ==================================================
    def _compute_response(self, xf, gf):
        """
        计算响应图 R_t = F^-1[Σ(z_k ⊙ g_k)]
        """
        response_f = np.sum(np.conj(gf) * xf, axis=2)
        response = np.real(np.fft.ifft2(response_f))
        return response

    # ==================================================
    # 响应变化计算（论文公式2）
    # ==================================================
    def _compute_response_variation(self, R_t, R_prev, max_pos_t, max_pos_prev):
        """
        计算局部和全局响应变化
        论文公式2: Π_i = |R_t[ψ_Δ]_i - R_{t-1}_i| / |R_{t-1}_i|
        """
        if R_prev is None:
            return np.zeros_like(R_t), 0.0

        # 对齐峰值位置（移除运动影响）
        shift_y = max_pos_prev[0] - max_pos_t[0]
        shift_x = max_pos_prev[1] - max_pos_t[1]

        # 使用np.roll进行循环移位
        R_t_shifted = np.roll(R_t, (shift_y, shift_x), axis=(0, 1))

        # 计算局部变化（论文公式2）
        epsilon = 1e-6
        Pi_local = np.abs(R_t_shifted - R_prev) / (np.abs(R_prev) + epsilon)

        # 计算全局变化（L2范数）
        Pi_global = np.linalg.norm(Pi_local)

        return Pi_local, Pi_global

    # ==================================================
    # 自动空间正则化（论文公式3）
    # ==================================================
    def _compute_adaptive_spatial_regularization(self, Pi_local):
        """
        论文公式3: ũ = P⊤[δ log(Π + 1) + u]
        其中P⊤用于裁剪中心区域，δ调整权重
        """
        # 对数变换局部变化
        delta_term = self.delta * np.log(Pi_local + 1)

        # 平滑处理（减少噪声）
        delta_term = cv2.GaussianBlur(delta_term, (5, 5), 1.0)

        # 组合基础空间正则化
        u_adaptive = delta_term + self.u_spatial

        # 裁剪到合理范围（避免过度惩罚）
        u_adaptive = np.clip(u_adaptive, 0.001, 100)

        return u_adaptive

    # ==================================================
    # 自动时间正则化（论文公式4）
    # ==================================================
    def _compute_temporal_regularization_ref(self, Pi_global):
        """
        论文公式4: θ̃ = ζ / (1 + log(ν||Π||_2 + 1))
        当全局变化大时，θ̃变小，允许滤波器快速适应
        """
        if Pi_global > self.phi:
            # 异常情况，返回极大值（停止学习）
            return 1e6

        # 论文公式4
        theta_ref = self.zeta / (1 + np.log(self.nu * Pi_global + 1))

        return theta_ref

    # ==================================================
    # ADMM优化训练（论文公式5-15）
    # ==================================================
    def _train_admm(self, xf, is_first_frame=False):
        """
        使用ADMM求解论文公式5:
        E(H_t, θ_t) = 1/2||y - Σx_k⊛h_k||² + 1/2Σ||ũ⊙h_k||²
                      + θ_t/2Σ||h_k - h_{k,t-1}||² + 1/2||θ_t - θ̃||²
        """
        H, W, K = xf.shape
        T = H * W

        # 预计算 S_xx = Σ|x_k|² （所有通道的功率谱）
        S_xx = np.sum(np.real(np.conj(xf) * xf), axis=2)

        # 重置ADMM参数
        gamma = self.gamma
        theta_t = self.theta_t

        # ADMM迭代（论文Section 4.2）
        for iter_idx in range(self.admm_iter):

            # ========== 子问题1: 更新 G（论文公式9-11）==========
            # 使用Sherman-Morrison公式的简化版本
            for k in range(K):
                # 计算分子
                numerator = (
                        np.conj(xf[:, :, k]) * self.yf / T +
                        gamma * (self.h_f[:, :, k] - self.m_f[:, :, k] / gamma)
                )

                if not is_first_frame:
                    numerator += theta_t * self.g_f_prev[:, :, k]

                # 计算分母
                denominator = (
                        np.abs(xf[:, :, k]) ** 2 + gamma +
                        (theta_t if not is_first_frame else 0)
                )

                # 更新G
                self.g_f[:, :, k] = numerator / (denominator + 1e-8)

            # ========== 子问题2: 更新 H（论文公式12-13）==========
            # 转到空间域
            g_plus_m = self.g_f + self.m_f / gamma
            g_plus_m_spatial = np.real(np.fft.ifft2(g_plus_m, axes=(0, 1)))

            # 应用空间正则化
            denominator_spatial = (
                    self.lambda_reg * self.u_tilde[:, :, None] ** 2 + gamma * T
            )
            h_spatial = (gamma * T * g_plus_m_spatial) / denominator_spatial

            # 转回频域
            self.h_f = np.fft.fft2(h_spatial, axes=(0, 1))

            # ========== 子问题3: 更新 θ_t（论文公式14）==========
            if not is_first_frame:
                # 计算滤波器变化
                diff_norm_sq = np.sum(np.abs(self.g_f - self.g_f_prev) ** 2)

                # 论文公式14: θ*_t = θ̃ - Σ||ĝ_k - ĝ_{k,t-1}||²/2
                theta_t = max(0, self.theta_ref - diff_norm_sq / 2)

            # ========== 更新拉格朗日乘子（论文公式15）==========
            self.m_f += gamma * (self.g_f - self.h_f)

            # 更新gamma（论文中的γ^(i+1) = min(γ_max, βγ^i)）
            gamma = min(self.gamma_max, self.gamma_step * gamma)

        # 保存当前滤波器作为下一帧的历史
        self.g_f_prev = self.g_f.copy()
        self.theta_t = theta_t

        if not is_first_frame:
            print(f"[ADMM] θ_ref={self.theta_ref:.2f}, θ_t={theta_t:.2f}")

    # ==================================================
    # 工具函数
    # ==================================================
    def _gaussian_label(self, sz, sigma):
        """创建高斯标签"""
        h, w = sz
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        label = np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2) / sigma ** 2)
        return label

    def _init_spatial_regularization(self, sz):
        """
        初始化碗形空间正则化（基于SRDCF）
        论文公式3中的u，用于减少边界效应
        """
        h, w = sz
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2

        # 碗形函数：中心区域正则化小，边缘大
        max_dist = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        u = 0.5 * (dist / max_dist)  # 归一化到[0, 0.5]

        return u


# ==================================================
# 主程序
# ==================================================
if __name__ == '__main__':
    print("=" * 60)
    print("AutoTrack: 自动时空正则化视觉跟踪器")
    print("论文: CVPR 2020")
    print("=" * 60)

    # 打开视频
    video_source = "./video/1.mp4"  # 修改为你的视频路径
    # video_source = 0  # 使用摄像头

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {video_source}")
        exit(1)

    ret, frame = cap.read()
    if not ret:
        print("错误: 无法读取第一帧")
        exit(1)

    # 选择目标
    print("\n请在窗口中框选目标...")
    bbox = cv2.selectROI("AutoTrack - 选择目标", frame, False, False)
    cv2.destroyWindow("AutoTrack - 选择目标")

    if bbox[2] == 0 or bbox[3] == 0:
        print("错误: 未选择有效目标")
        exit(1)

    # 初始化跟踪器
    print("\n初始化跟踪器...")
    tracker = AutoTrack()
    tracker.init(frame, bbox)

    print("\n开始跟踪...")
    print("按 ESC 退出，按 SPACE 暂停")

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n视频结束")
                break

            frame_idx += 1

            # 跟踪
            pos, target_sz = tracker.track(frame)

            # 绘制结果
            cy, cx = pos.astype(int)
            h, w = target_sz.astype(int)

            # 边界框
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 中心点
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            # 信息文本
            cv2.putText(frame, f"AutoTrack", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 显示
        cv2.imshow("AutoTrack Tracking", frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
            print(f"{'暂停' if paused else '继续'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n跟踪结束")