"""
AutoTrack vs AutoTrack4 对比测试脚本
在同一视频中同时显示两个跟踪器的跟踪效果
- AutoTrack: 绿色框
- AutoTrack4: 蓝色框（正常）/ 红色框（遮挡时Kalman权重增大）
实时显示PSR曲线、尺度曲线和响应热力图
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 设置图表字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# 导入两个跟踪器
from AutoTrack import AutoTrack
from AutoTrack_improved4 import AutoTrack4


def compare_trackers(video_path, bbox=None):
    """
    对比两个跟踪器，实时显示图表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("无法读取视频第一帧")
        return

    # 选择初始边界框
    if bbox is None:
        bbox = cv2.selectROI("Select Target", frame, False, False)
        cv2.destroyWindow("Select Target")

    print(f"初始边界框: {bbox}")

    # 初始化两个跟踪器
    tracker1 = AutoTrack()
    tracker1.init(frame.copy(), bbox)

    tracker2 = AutoTrack4()
    tracker2.init(frame.copy(), bbox)

    # 记录历史数据
    frame_idx = 0
    pos_history_1 = []
    pos_history_2 = []
    psr_history_1 = []
    psr_history_2 = []
    scale_history_2 = []
    occlusion_history = []

    # 计算 AutoTrack 的 PSR
    def compute_psr_auto(response, peak_size=5):
        H, W = response.shape
        py, px = np.unravel_index(np.argmax(response), response.shape)
        half = peak_size // 2
        y1, y2 = max(0, py - half), min(H, py + half + 1)
        x1, x2 = max(0, px - half), min(W, px + half + 1)
        side = response.copy()
        side[y1:y2, x1:x2] = 0
        mean = np.mean(side)
        std = np.std(side) + 1e-6
        return (response[py, px] - mean) / std

    print("\n开始跟踪...")
    print("=" * 60)

    # ========== 初始化实时图表 ==========
    plt.ion()

    # Figure 1: PSR曲线和尺度曲线
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
    fig1.suptitle('AutoTrack_Imp Tracking Metrics', fontsize=12)

    ax_psr = axes1[0]
    ax_psr.set_xlabel('Frame')
    ax_psr.set_ylabel('PSR')
    ax_psr.set_title('PSR over Time')
    #ax_psr.axhline(y=8.0, color='r', linestyle='--', label='Occlusion Threshold', linewidth=1)
    ax_psr.legend(loc='upper right')
    ax_psr.grid(True, alpha=0.3)
    line_psr, = ax_psr.plot([], [], 'b-', linewidth=1.5)
    scatter_occ = ax_psr.scatter([], [], c='r', s=15, label='Occluded', zorder=5)

    ax_scale = axes1[1]
    ax_scale.set_xlabel('Frame')
    ax_scale.set_ylabel('Scale Factor')
    ax_scale.set_title('Scale Factor over Time')
    ax_scale.grid(True, alpha=0.3)
    line_scale, = ax_scale.plot([], [], 'g-', linewidth=1.5)

    fig1.tight_layout()

    # Figure 2: 响应热力图
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle('Response Heatmaps', fontsize=12)

    ax_heat1 = axes2[0]
    ax_heat1.set_title('AutoTrack Response')
    heatmap1 = ax_heat1.imshow(np.zeros((tracker1.Hc, tracker1.Wc)),
                                cmap='jet', interpolation='nearest', origin='upper')
    cbar1 = plt.colorbar(heatmap1, ax=ax_heat1, fraction=0.046, pad=0.04)

    ax_heat2 = axes2[1]
    ax_heat2.set_title('AutoTrack_Imp Response')
    heatmap2 = ax_heat2.imshow(np.zeros((tracker2.Hc, tracker2.Wc)),
                                cmap='jet', interpolation='nearest', origin='upper')
    cbar2 = plt.colorbar(heatmap2, ax=ax_heat2, fraction=0.046, pad=0.04)

    fig2.tight_layout()

    plt.show()

    # ========== 跟踪循环 ==========
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # AutoTrack 跟踪
        pos1, target_sz1, response1 = tracker1.track(frame.copy())
        cy1, cx1 = pos1.astype(int)
        h1, w1 = target_sz1.astype(int)
        psr1 = compute_psr_auto(response1)

        # AutoTrack4 跟踪
        pos2, target_sz2, response2 = tracker2.track(frame.copy())
        cy2, cx2 = pos2.astype(int)
        h2, w2 = target_sz2.astype(int)
        psr2 = tracker2.psr

        # 保存历史
        pos_history_1.append((cx1, cy1))
        pos_history_2.append((cx2, cy2))
        psr_history_1.append(psr1)
        psr_history_2.append(psr2)
        scale_history_2.append(tracker2.current_scale_factor)

        # 判断遮挡状态
        is_occluded = psr2 < tracker2.psr_smooth_threshold
        occlusion_history.append(is_occluded)

        # 绘制跟踪框
        display = frame.copy()

        # AutoTrack - 绿色框
        cv2.rectangle(display,
                      (cx1 - w1 // 2, cy1 - h1 // 2),
                      (cx1 + w1 // 2, cy1 + h1 // 2),
                      (0, 255, 0), 2)
        cv2.putText(display, "AutoTrack", (cx1 - w1 // 2, cy1 - h1 // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # AutoTrack4 - 根据遮挡状态改变颜色
        if is_occluded:
            color2 = (0, 0, 255)  # 红色 - 遮挡
            status2 = "OCCLUDED"
        else:
            color2 = (255, 0, 0)  # 蓝色 - 正常
            status2 = "TRACKING"

        cv2.rectangle(display,
                      (cx2 - w2 // 2, cy2 - h2 // 2),
                      (cx2 + w2 // 2, cy2 + h2 // 2),
                      color2, 2)
        cv2.putText(display, f"AutoTrack_Imp [{status2}]", (cx2 - w2 // 2, cy2 - h2 // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

        # 显示信息
        info_y = 30
        cv2.putText(display, f"Frame: {frame_idx}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_y += 25
        cv2.putText(display, f"AutoTrack - PSR: {psr1:.2f}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        info_y += 20
        cv2.putText(display, f"AutoTrack_Imp - PSR: {psr2:.2f} | Scale: {tracker2.current_scale_factor:.3f}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

        center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        info_y += 20
        cv2.putText(display, f"Center Distance: {center_dist:.1f} px", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("AutoTrack vs AutoTrack_Imp", display)

        # ========== 更新实时图表 ==========
        frames = np.arange(frame_idx)

        # 更新PSR曲线
        line_psr.set_data(frames, psr_history_2)
        ax_psr.relim()
        ax_psr.autoscale_view()

        # 更新遮挡散点
        occ_idx = np.where(occlusion_history)[0]
        if len(occ_idx) > 0:
            scatter_occ.set_offsets(np.column_stack([occ_idx, np.array(psr_history_2)[occ_idx]]))

        # 更新尺度曲线
        line_scale.set_data(frames, scale_history_2)
        ax_scale.relim()
        ax_scale.autoscale_view()

        # 更新热力图
        heatmap1.set_data(response1)
        heatmap1.set_clim(vmin=response1.min(), vmax=response1.max())

        heatmap2.set_data(response2)
        heatmap2.set_clim(vmin=response2.min(), vmax=response2.max())

        # 刷新图表
        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()
        plt.pause(0.001)

        # 按键控制
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # 空格
            cv2.waitKey(0)

    # ========== 结束处理 ==========
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()

    # 保存图表
    fig1.savefig('tracker_comparison.png', dpi=150, bbox_inches='tight')
    fig2.savefig('response_heatmaps.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: tracker_comparison.png, response_heatmaps.png")

    # 打印统计信息
    print_statistics(psr_history_1, psr_history_2, scale_history_2,
                     pos_history_1, pos_history_2, occlusion_history)

    plt.show()


def print_statistics(psr1, psr2, scale2, pos1, pos2, occ_hist):
    """打印统计信息"""
    pos1_arr = np.array(pos1)
    pos2_arr = np.array(pos2)

    print("\n" + "=" * 60)
    print("跟踪统计信息")
    print("=" * 60)

    print(f"\n总帧数: {len(psr1)}")

    print("\n--- AutoTrack ---")
    print(f"平均 PSR: {np.mean(psr1):.2f}")
    print(f"PSR 标准差: {np.std(psr1):.2f}")
    print(f"最小 PSR: {np.min(psr1):.2f}")
    print(f"最大 PSR: {np.max(psr1):.2f}")

    print("\n--- AutoTrack_Imp ---")
    print(f"平均 PSR: {np.mean(psr2):.2f}")
    print(f"PSR 标准差: {np.std(psr2):.2f}")
    print(f"最小 PSR: {np.min(psr2):.2f}")
    print(f"最大 PSR: {np.max(psr2):.2f}")
    print(f"平均尺度: {np.mean(scale2):.3f}")
    print(f"尺度范围: [{np.min(scale2):.3f}, {np.max(scale2):.3f}]")

    occ_count = np.sum(occ_hist)
    print(f"\n遮挡帧数: {occ_count} ({100*occ_count/len(occ_hist):.1f}%)")

    # 中心点距离统计
    center_dist = np.sqrt(np.sum((pos1_arr - pos2_arr)**2, axis=1))
    print("\n--- 位置差异 ---")
    print(f"平均中心距离: {np.mean(center_dist):.2f} 像素")
    print(f"最大中心距离: {np.max(center_dist):.2f} 像素")

    print("=" * 60)


if __name__ == '__main__':
    video_path = "./video/output2.mp4"
    compare_trackers(video_path)
