"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
# python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
# local libs
from uqim_utils import getUIQM
import argparse


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch FUnIE-GAN UIQM Metric Runner")
    parser.add_argument("-d", "--data", default=r"E:\Underwater_Image_Enhancement\Our_Dataset\Our_results", type=str, metavar="PATH",
                        help="path to images (default: none)")

    args = parser.parse_args()

    uqims = measure_UIQMs(args.data)
    print(f"[UIQM] Mean={np.mean(uqims):.3f}, Std={np.std(uqims):.3f}")

    # Mean=2.990, Std=0.498


# OUR UIEB PAIR TEST
# [UIQM] Mean=3.248, Std=0.427

# rghs uieb pair test
# [UIQM] Mean=2.698, Std=0.394

# funie gan uieb pair 
# [UIQM] Mean=2.887, Std=0.338

# puie uieb pair test
# [UIQM] Mean=2.859, Std=0.413

# ruenet uibe pair test
# [UIQM] Mean=2.827, Std=0.447

# shallowuwnet pair uieb test
# [UIQM] Mean=2.976, Std=0.288

# ulap uieb test 
# [UIQM] Mean=2.101, Std=0.459

# uwcyclegan uieb pair test
# [UIQM] Mean=1.997, Std=0.327

# waternet uieb pair test
# [UIQM] Mean=3.106, Std=0.279



# Ours U45
# [UIQM] Mean=3.600, Std=0.347

# FUNIEGAN U45
# [UIQM] Mean=3.190, Std=0.312

# uwcyclegan u45
# [UIQM] Mean=3.412, Std=0.177

# waternet u45
# [UIQM] Mean=3.175, Std=0.117

# ULAP U45
# [UIQM] Mean=2.977, Std=0.285

# RUENET U45
# [UIQM] Mean=3.399, Std=0.296

# RGHS U45
# [UIQM] Mean=3.352, Std=0.104

# puie u45
# [UIQM] Mean=3.599, Std=0.110

# shallow u45
# [UIQM] Mean=3.103, Std=0.343


# Our dataset
# UIQM] Mean=1.631, Std=0.533