# -*- coding: utf-8 -*-
"""
批量处理图像，使用 Dullrazor 算法去除图像中的毛发
保持源文件夹的子目录结构
"""
import argparse
import os
from glob import glob

from dullrazor import apply_dullrazor_image

import cv2

def should_skip(path: str, skip_keywords: list[str]) -> bool:
    lowered = path.lower()
    return any(keyword in lowered for keyword in skip_keywords)


def batch_process_images(input_dir, output_dir, skip_keywords):
    """
    批量处理图像，保持子目录结构
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    # 查找所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        # 搜索根目录中的图像
        image_files.extend(glob(os.path.join(input_dir, ext)))
        # 搜索所有子目录中的图像
        image_files.extend(glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    print(f"找到 {len(image_files)} 个图像文件")
    if not image_files:
        raise RuntimeError(f"未找到可处理的图像文件: {input_dir}")
    
    processed_count = 0
    
    for i, image_path in enumerate(image_files):
        if should_skip(image_path, skip_keywords):
            continue
        print(f"正在处理 ({i+1}/{len(image_files)}): {os.path.relpath(image_path, input_dir)}")
        
        # 应用 Dullrazor 算法
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        processed_image = apply_dullrazor_image(image)
        
        if processed_image is not None:
            # 计算相对于输入目录的路径
            rel_path = os.path.relpath(image_path, input_dir)
            # 构建输出路径
            output_path = os.path.join(output_dir, rel_path)
            
            # 确保输出目录存在
            output_folder = os.path.dirname(output_path)
            os.makedirs(output_folder, exist_ok=True)
            
            # 保存处理后的图像
            success = cv2.imwrite(output_path, processed_image)
            
            if success:
                print(f"已保存: {output_path}")
                processed_count += 1
            else:
                print(f"保存失败: {output_path}")
        else:
            print(f"处理失败: {image_path}")
    
    print(f"\n完成! 成功处理了 {processed_count} / {len(image_files)} 个图像")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量去毛处理（DullRazor）。")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/root/autodl-tmp/data/HAM10000/input",
        help="HAM10000 输入根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认原地覆盖）",
    )
    parser.add_argument(
        "--skip-keywords",
        type=str,
        default="seg,mask,segmentation",
        help="跳过文件的关键字（逗号分隔）",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_directory = args.input_dir
    output_directory = args.output_dir or input_directory
    skip_keywords = [k.strip() for k in args.skip_keywords.split(",") if k.strip()]

    batch_process_images(input_directory, output_directory, skip_keywords)