import cv2
import os

image_dir = "./img"       # 图片文件夹
output_video = "output.mp4"
fps = 30                   # 视频帧率

# 读取所有图片并排序
images = sorted([
    img for img in os.listdir(image_dir)
    if img.endswith(".png") or img.endswith(".jpg")
])

# 读第一张，获取尺寸
first_frame = cv2.imread(os.path.join(image_dir, images[0]))
height, width, _ = first_frame.shape

# 定义视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 生成 mp4
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for img_name in images:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print("视频生成完成")
