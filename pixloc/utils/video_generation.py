import cv2
import os
import re


def _make_image_sort_key(img_path, mode="auto"):
    stem = os.path.basename(img_path).rsplit(".", 1)[0]
    mode = (mode or "auto").lower()

    if mode != "timestamp":
        match = re.fullmatch(r"(\d+)_(\d+)", stem)
        if match:
            return (0, int(match.group(1)), int(match.group(2)), stem)

    if mode != "index" and re.fullmatch(r"\d+(?:\.\d+)?", stem):
        return (1, float(stem), stem)

    return (2, stem)


def create_video_from_images(image_folder, output_video_path, image_size=(512, 288), fps=25, sort_mode="auto"):
    """
    从文件夹中的图像创建视频
    :param image_folder: 包含图像的文件夹路径
    :param output_video_path: 输出视频的路径
    :param image_size: 输出视频的分辨率，默认为(400, 300)
    :param fps: 输出视频的帧率，默认为24
    """
    # 获取文件夹中所有图像文件
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
              if img.endswith(".jpg") or img.endswith(".png")]  # 支持.jpg和.png格式

    # 按自定义排序函数排序
    images.sort(key=lambda p: _make_image_sort_key(p, sort_mode))

    # 检查是否有图像文件
    if not images:
        print("未找到符合条件的图像文件！")
        return

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, image_size)

    # 遍历图像文件
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像：{img_path}")
            continue

        # 调整图像大小以匹配视频分辨率
        img_resized = cv2.resize(img, image_size)

        # 写入视频
        video_writer.write(img_resized)

    # 释放视频写入器
    video_writer.release()
    print(f"视频已生成：{output_video_path}")
