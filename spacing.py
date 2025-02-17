import os
import numpy as np
import SimpleITK as sitk

input_path = "/data4/liruocheng/dataset/others/images"
output_path = "/data4/liruocheng/dataset/others/images_256"

os.makedirs(output_path, exist_ok=True)


def resample_image(image):
    """
    通过SimpleITK对整个3D图像进行重采样。

    Parameters:
    image (sitk.Image): 输入的SimpleITK图像对象
    new_size (tuple): 目标尺寸 (z, y, x)

    Returns:
    sitk.Image: 重采样后的SimpleITK图像
    """
    original_size = image.GetSize()  # 当前图像的尺寸
    original_spacing = image.GetSpacing()  # 当前图像的spacing
    new_size = ( 256, 256, original_size[2])

    # 计算新spacing来匹配目标尺寸
    new_spacing = [
        original_size[i] * original_spacing[i] / new_size[i] for i in range(3)
    ]

    # 定义重采样器
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)  # 设置输出的spacing
    resample.SetSize(new_size)  # 设置输出尺寸
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  # 使用三次样条插值
    resample.SetOutputDirection(image.GetDirection())  # 保持方向
    resample.SetOutputOrigin(image.GetOrigin())  # 保持原点

    # 执行重采样
    new_image = resample.Execute(image)
    return new_image


def process_file(file_path, output_path):
    """
    处理单个文件，将其重采样为目标大小并保存。

    Parameters:
    file_path (str): 输入文件的路径
    output_path (str): 输出文件夹的路径
    """
    img = sitk.ReadImage(file_path)

    # 进行重采样
    new_img = resample_image(img)

    # 保存图像
    filename = os.path.basename(file_path)
    sitk.WriteImage(new_img, os.path.join(output_path, filename))
    print(f"{filename} 处理完成")


# 处理所有文件
for filename in os.listdir(input_path):
    if filename.endswith(".nii.gz"):
        file_path = os.path.join(input_path, filename)
        print(f"Processing {filename}...")
        process_file(file_path, output_path)

print("所有文件处理完成！")