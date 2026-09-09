"""提供与固定 BasicSR RGB 口径等价的 PSNR 和 SSIM 实现。"""

import cv2
import numpy as np


def reorder_image(image, input_order="HWC"):
    """将二维、HWC 或 CHW 图像统一转换为 HWC。

    Args:
        image: 输入图像，支持二维、HWC 或 CHW NumPy 数组。
        input_order: 三维输入的排列方式，只允许 ``HWC`` 或 ``CHW``。

    Returns:
        按 HWC 排列的 NumPy 数组；二维输入会补充单通道维度。

    Raises:
        ValueError: ``input_order`` 不受支持。
    """
    if input_order not in ("HWC", "CHW"):
        raise ValueError(
            f"不支持 input_order={input_order!r}，只允许 'HWC' 或 'CHW'。"
        )

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    if input_order == "CHW":
        image = image.transpose(1, 2, 0)
    return image


def _prepare_rgb_pair(img1, img2, crop_border, input_order, test_y_channel):
    """校验并准备一对 RGB `[0,255]` 图像。

    Args:
        img1: 第一张 RGB 图像。
        img2: 第二张 RGB 图像。
        crop_border: 每侧需要排除的边界像素数。
        input_order: 输入图像的排列方式。
        test_y_channel: 兼容 BasicSR 的参数；固定口径只允许 ``False``。

    Returns:
        转为 HWC、float64 并完成边界裁剪的图像对。

    Raises:
        ValueError: 输入不符合固定 RGB 评价口径。
    """
    if test_y_channel:
        raise ValueError("统一口径固定使用 RGB，test_y_channel 必须为 False。")
    if not isinstance(crop_border, int) or crop_border < 0:
        raise ValueError("crop_border 必须是非负整数。")

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不一致：{img1.shape} 与 {img2.shape}。")
    if img1.ndim != 3 or img1.shape[2] != 3:
        raise ValueError(f"统一口径要求三通道 RGB 图像，当前尺寸为 {img1.shape}。")

    height, width = img1.shape[:2]
    if crop_border * 2 >= height or crop_border * 2 >= width:
        raise ValueError(
            f"crop_border={crop_border} 对当前图像尺寸 {height}x{width} 过大。"
        )

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    return img1, img2


def calculate_psnr(
    img1,
    img2,
    crop_border=0,
    input_order="HWC",
    test_y_channel=False,
):
    """按 BasicSR RGB 联合 MSE 口径计算一张图像的 PSNR。

    Args:
        img1: 第一张 RGB 图像，数值范围固定为 `[0,255]`。
        img2: 第二张 RGB 图像，数值范围固定为 `[0,255]`。
        crop_border: 每侧排除的边界像素数；统一实验固定为 ``0``。
        input_order: 输入排列方式，支持 ``HWC`` 或 ``CHW``。
        test_y_channel: 兼容 BasicSR 的参数；统一实验固定为 ``False``。

    Returns:
        单张图像的 PSNR；两张图完全相同时返回正无穷。
    """
    img1, img2 = _prepare_rgb_pair(
        img1, img2, crop_border, input_order, test_y_channel
    )

    # BasicSR RGB PSNR 对 H、W、C 的全部元素共同计算一个 MSE。
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim_single_channel(img1, img2):
    """按 BasicSR 的 MATLAB-like 实现计算单通道 SSIM。

    Args:
        img1: 第一张 `[0,255]` 单通道 float64 图像。
        img2: 第二张 `[0,255]` 单通道 float64 图像。

    Returns:
        单通道 valid 区域 SSIM map 的算术平均值。
    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # 与 BasicSR 一致，卷积后排除 Gaussian 窗口影响的 5 像素边缘。
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = (
        cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    )

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((numerator / denominator).mean())


def calculate_ssim(
    img1,
    img2,
    crop_border=0,
    input_order="HWC",
    test_y_channel=False,
):
    """按 BasicSR RGB 逐通道平均口径计算一张图像的 SSIM。

    Args:
        img1: 第一张 RGB 图像，数值范围固定为 `[0,255]`。
        img2: 第二张 RGB 图像，数值范围固定为 `[0,255]`。
        crop_border: 每侧排除的边界像素数；统一实验固定为 ``0``。
        input_order: 输入排列方式，支持 ``HWC`` 或 ``CHW``。
        test_y_channel: 兼容 BasicSR 的参数；统一实验固定为 ``False``。

    Returns:
        R、G、B 三个单通道 SSIM 的算术平均值。

    Raises:
        ValueError: 裁剪后的图像小于 `11x11`，无法使用固定 Gaussian 窗口。
    """
    img1, img2 = _prepare_rgb_pair(
        img1, img2, crop_border, input_order, test_y_channel
    )
    if img1.shape[0] < 11 or img1.shape[1] < 11:
        raise ValueError(
            f"SSIM 要求裁剪后的图像至少为 11x11，当前为 {img1.shape[:2]}。"
        )

    channel_scores = [
        _ssim_single_channel(img1[..., channel], img2[..., channel])
        for channel in range(3)
    ]
    return float(np.mean(channel_scores))
