import os
import argparse
import logging
import cv2
import numpy as np
from os import path as osp

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils import get_root_logger, get_time_str


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    os.makedirs(args.log_path, exist_ok=True)
    log_file = osp.join(args.log_path, f"{args.model_name}_SPMCs_{get_time_str()}.log")
    logger = get_root_logger(logger_name='SPMCs', log_level=logging.INFO, log_file=log_file)

    psnr_all = []
    ssim_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_restored = sorted(list(scandir(args.restored, recursive=True, full_path=True)))

    if args.test_y_channel:
        logger.info('Testing Y channel.')
    else:
        logger.info('Testing RGB channels.')

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if args.suffix == '':
            img_path_restored = img_list_restored[i]
        else:
            img_path_restored = osp.join(args.restored, basename + args.suffix + ext)
        img_restored = cv2.imread(img_path_restored, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

        print(f"gt_path : {img_path} sr_path : {img_path_restored}")

        if args.correct_mean_var:
            mean_l = []
            std_l = []
            for j in range(3):
                mean_l.append(np.mean(img_gt[:, :, j]))
                std_l.append(np.std(img_gt[:, :, j]))
            for j in range(3):
                # correct twice
                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                mean = np.mean(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] - mean + mean_l[j]
                std = np.std(img_restored[:, :, j])
                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

        if args.test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)

        # Resize GT
        if img_gt.shape != img_restored.shape:           
            img_gt = img_gt[0:img_restored.shape[0], 0:img_restored.shape[1] ]
        # calculate PSNR and SSIM
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        logger.info(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    logger.info(args.gt)
    logger.info(args.restored)
    logger.info(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BasicVSR', help='method name')
    parser.add_argument('--gt', type=str, default='datasets/val_set14/Set14', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results/Set14', help='Path to restored images')
    parser.add_argument('--log_path', type=str, default='results/SPMCs', help='Path to log file')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument(
        '--test_y_channel',
        action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--correct_mean_var', action='store_true', help='Correct the mean and var of restored images.')
    args = parser.parse_args()
    main(args)
