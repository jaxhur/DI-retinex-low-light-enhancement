"""DI-Retinex 测试阶段的全参考图像质量评价工具。"""

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from unified_image_metrics import calculate_psnr, calculate_ssim


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def load_rgb_tensor(image_path, device):
    """读取 RGB 图像并返回范围为 [0, 1] 的单张 GPU Tensor。"""
    image = Image.open(image_path).convert('RGB')
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _build_image_index(root_path):
    """以规范化相对路径建立图像索引，避免同名文件被错误配对。"""
    root = Path(root_path)
    if not root.is_dir():
        raise FileNotFoundError('图像目录不存在: {}'.format(root))

    image_index = {}
    for image_path in sorted(root.rglob('*')):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_key = image_path.relative_to(root).as_posix()
        if relative_key in image_index:
            raise RuntimeError('发现重复图像相对路径: {}'.format(relative_key))
        image_index[relative_key] = image_path

    if not image_index:
        raise RuntimeError('图像目录中没有可评测图像: {}'.format(root))
    return image_index


def build_test_pairs(lowlight_root, gt_root=None):
    """构造低照图与 GT 的一一配对；未提供 GT 时只返回低照图列表。"""
    lowlight_index = _build_image_index(lowlight_root)
    if gt_root is None:
        return [(key, path, None) for key, path in sorted(lowlight_index.items())]

    gt_index = _build_image_index(gt_root)
    missing_gt = sorted(set(lowlight_index) - set(gt_index))
    missing_lowlight = sorted(set(gt_index) - set(lowlight_index))
    if missing_gt or missing_lowlight:
        messages = []
        if missing_gt:
            messages.append('GT 缺少 {} 张，示例: {}'.format(len(missing_gt), missing_gt[:3]))
        if missing_lowlight:
            messages.append('低照图缺少 {} 张，示例: {}'.format(len(missing_lowlight), missing_lowlight[:3]))
        raise RuntimeError('LQ/GT 配对失败；' + '；'.join(messages))

    return [
        (key, lowlight_index[key], gt_index[key])
        for key in sorted(lowlight_index)
    ]


class FullReferenceMetrics:
    """按统一复现口径计算 RGB PSNR、RGB SSIM 与 LPIPS-Alex-v0.1。"""

    def __init__(self, device):
        """初始化 LPIPS 模型；device 与增强网络保持一致。"""
        try:
            import lpips
        except ImportError as error:
            raise ImportError(
                '计算 LPIPS 需要 lpips；请执行 pip install lpips。'
            ) from error

        self.device = device
        self.calculate_psnr = calculate_psnr
        self.calculate_ssim = calculate_ssim
        self.lpips_model = lpips.LPIPS(net='alex', version='0.1').to(device).eval()

    def calculate(self, enhanced, gt):
        """计算同尺寸增强图与 GT 的单图全参考指标。"""
        if enhanced.shape != gt.shape:
            raise ValueError(
                '增强图与 GT 尺寸不一致: enhanced={}, gt={}'.format(
                    tuple(enhanced.shape), tuple(gt.shape)
                )
            )

        # 不 resize、不做 GT-Mean，直接在原始空间尺寸上评测。
        enhanced = enhanced.detach().clamp(0, 1)
        gt = gt.detach().clamp(0, 1)
        enhanced_hwc = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        gt_hwc = gt.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0

        psnr = self.calculate_psnr(
            gt_hwc,
            enhanced_hwc,
            crop_border=0,
            input_order='HWC',
            test_y_channel=False,
        )
        rgb_ssim = self.calculate_ssim(
            gt_hwc,
            enhanced_hwc,
            crop_border=0,
            input_order='HWC',
            test_y_channel=False,
        )

        # LPIPS 要求 RGB Tensor 位于 [-1, 1]，且使用 AlexNet v0.1。
        with torch.no_grad():
            lpips_value = self.lpips_model(enhanced * 2 - 1, gt * 2 - 1).item()
        return {'psnr': float(psnr), 'rgb_ssim': float(rgb_ssim), 'lpips': float(lpips_value)}


def calculate_mean_metrics(metric_values):
    """对完整测试集的逐图指标做算术平均。"""
    if not metric_values:
        raise ValueError('没有可用于汇总的指标。')
    return {
        'psnr': float(np.mean([item['psnr'] for item in metric_values])),
        'rgb_ssim': float(np.mean([item['rgb_ssim'] for item in metric_values])),
        'lpips': float(np.mean([item['lpips'] for item in metric_values])),
    }


def write_metric_csv(csv_path, experiment_name, dataset_name, metrics, image_count,
                     checkpoint_path, enhanced_images_path):
    """写入单行测试集平均指标，便于后续汇总到复现实验账本。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'experiment', 'dataset', 'image_count', 'psnr', 'ssim', 'ssim_mode', 'lpips',
        'lpips_backbone', 'lpips_version', 'checkpoint', 'enhanced_images', 'metric_source'
    ]
    row = {
        'experiment': experiment_name,
        'dataset': dataset_name,
        'image_count': image_count,
        'psnr': '{:.4f}'.format(metrics['psnr']),
        'ssim': '{:.4f}'.format(metrics['rgb_ssim']),
        'ssim_mode': 'RGB',
        'lpips': '{:.4f}'.format(metrics['lpips']),
        'lpips_backbone': 'alex',
        'lpips_version': '0.1',
        'checkpoint': os.path.abspath(checkpoint_path),
        'enhanced_images': os.path.abspath(enhanced_images_path),
        'metric_source': 'basicsr_equivalent_rgb_lpips_alex_v0.1',
    }
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
