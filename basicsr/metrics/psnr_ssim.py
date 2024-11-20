import cv2
import numpy as np
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor

import os
import sys
from contextlib import redirect_stdout

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')

# def setup_lpips_silently():
#     # Suppress stdout temporarily
#     with open(os.devnull, 'w') as f, redirect_stdout(f):
#         loss_fn_vgg = lpips.LPIPS(net='vgg')
#     return loss_fn_vgg

def modcrop(img, scale):
    """Crop the image so that its dimensions are divisible by the scaling factor."""
    if img.ndim == 2:  # Grayscale
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:  # RGB
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError(f"Unsupported image dimensions: {img.ndim}")
    return img


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, scale=2, **kwargs):
    img = modcrop(img, scale)
    img2 = modcrop(img2, scale)

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW".')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, scale=2, **kwargs):
    img = modcrop(img, scale)
    img2 = modcrop(img2, scale)

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW".')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    def _ssim(img, img2):
        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map.mean()

    ssims = [_ssim(img[..., i], img2[..., i]) for i in range(img.shape[2])]
    return np.array(ssims).mean()


# @METRIC_REGISTRY.register()
# def calculate_lpips(img, img2, crop_border=0, net='vgg', input_order='HWC', scale=2, **kwargs):
#     """Calculate LPIPS (Learned Perceptual Image Patch Similarity)."""
#     img = modcrop(img, scale)
#     img2 = modcrop(img2, scale)

#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#     if input_order != 'HWC':
#         raise ValueError('LPIPS currently supports input_order="HWC" only.')

#     # Normalize images to [0, 1] and convert to tensors
#     img = img.astype(np.float32) / 255.
#     img2 = img2.astype(np.float32) / 255.

#     mean = [0.5, 0.5, 0.5]
#     std = [0.5, 0.5, 0.5]
#     img_tensor, img2_tensor = img2tensor([img, img2], bgr2rgb=True, float32=True)
#     normalize(img_tensor, mean, std, inplace=True)
#     normalize(img2_tensor, mean, std, inplace=True)

#     # Load LPIPS model
#     loss_fn_vgg = lpips.LPIPS(net=net).cuda()
#     lpips_val = loss_fn_vgg(img_tensor.unsqueeze(0).cuda(), img2_tensor.unsqueeze(0).cuda())
#     return lpips_val.item()

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border=0, net='vgg', input_order='HWC', scale=2, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)."""
    img = modcrop(img, scale)
    img2 = modcrop(img2, scale)

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order != 'HWC':
        raise ValueError('LPIPS currently supports input_order="HWC" only.')

    # Normalize images to [0, 1] and convert to tensors
    img = img.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_tensor, img2_tensor = img2tensor([img, img2], bgr2rgb=True, float32=True)
    normalize(img_tensor, mean, std, inplace=True)
    normalize(img2_tensor, mean, std, inplace=True)

    # def setup_lpips_silently():
    #     # Suppress stdout temporarily
    #     with open(os.devnull, 'w') as f, redirect_stdout(f):
    #         loss_fn_vgg = lpips.LPIPS(net='vgg')
    #     return loss_fn_vgg

    # # Load LPIPS model
    # # loss_fn_vgg = lpips.LPIPS(net=net).cuda()
    # # Setup LPIPS silently
    # loss_fn_vgg = setup_lpips_silently()
    # lpips_val = loss_fn_vgg(img_tensor.unsqueeze(0).cuda(), img2_tensor.unsqueeze(0).cuda())

    def setup_lpips_silently():
        """Setup LPIPS model silently."""
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda')  # Ensure LPIPS model is on GPU
        return loss_fn_vgg

    # Load LPIPS model
    loss_fn_vgg = setup_lpips_silently()

    # Ensure img_tensor and img2_tensor are on the same device
    img_tensor = img_tensor.to('cuda')
    img2_tensor = img2_tensor.to('cuda')

    # Calculate LPIPS
    lpips_val = loss_fn_vgg(img_tensor.unsqueeze(0), img2_tensor.unsqueeze(0))
    return lpips_val.item()

# import cv2
# import numpy as np

# from basicsr.metrics.metric_util import reorder_image, to_y_channel
# from basicsr.utils.registry import METRIC_REGISTRY


# @METRIC_REGISTRY.register()
# def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
#     """Calculate PSNR (Peak Signal-to-Noise Ratio).

#     Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the PSNR calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

#     Returns:
#         float: psnr result.
#     """

#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
#     img = reorder_image(img, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)

#     if crop_border != 0:
#         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

#     if test_y_channel:
#         img = to_y_channel(img)
#         img2 = to_y_channel(img2)

#     mse = np.mean((img - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 20. * np.log10(255. / np.sqrt(mse))


# def _ssim(img, img2):
#     """Calculate SSIM (structural similarity) for one channel images.

#     It is called by func:`calculate_ssim`.

#     Args:
#         img (ndarray): Images with range [0, 255] with order 'HWC'.
#         img2 (ndarray): Images with range [0, 255] with order 'HWC'.

#     Returns:
#         float: ssim result.
#     """

#     c1 = (0.01 * 255)**2
#     c2 = (0.03 * 255)**2

#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
#     return ssim_map.mean()


# @METRIC_REGISTRY.register()
# def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
#     """Calculate SSIM (structural similarity).

#     Ref:
#     Image quality assessment: From error visibility to structural similarity

#     The results are the same as that of the official released MATLAB code in
#     https://ece.uwaterloo.ca/~z70wang/research/ssim/.

#     For three-channel images, SSIM is calculated for each channel and then
#     averaged.

#     Args:
#         img (ndarray): Images with range [0, 255].
#         img2 (ndarray): Images with range [0, 255].
#         crop_border (int): Cropped pixels in each edge of an image. These
#             pixels are not involved in the SSIM calculation.
#         input_order (str): Whether the input order is 'HWC' or 'CHW'.
#             Default: 'HWC'.
#         test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

#     Returns:
#         float: ssim result.
#     """

#     assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
#     if input_order not in ['HWC', 'CHW']:
#         raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
#     img = reorder_image(img, input_order=input_order)
#     img2 = reorder_image(img2, input_order=input_order)
#     img = img.astype(np.float64)
#     img2 = img2.astype(np.float64)

#     if crop_border != 0:
#         img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
#         img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

#     if test_y_channel:
#         img = to_y_channel(img)
#         img2 = to_y_channel(img2)

#     ssims = []
#     for i in range(img.shape[2]):
#         ssims.append(_ssim(img[..., i], img2[..., i]))
#     return np.array(ssims).mean()
