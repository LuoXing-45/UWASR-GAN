"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM) 
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
# python libs
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
# local libs
from imqual_utils import getSSIM, getPSNR
import argparse


# compares avg ssim and psnr
def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    gtr_paths = glob(join(gtr_dir, "*.*"))
    gen_paths = glob(join(gen_dir, "*.*"))
    
    # 使用文件名（不带扩展名）作为键创建字典
    gtr_dict = {basename(p).split('.')[0]: p for p in gtr_paths}
    gen_dict = {basename(p).split('.')[0]: p for p in gen_paths}
    
    # 获取共同的文件名
    common_keys = set(gtr_dict.keys()) & set(gen_dict.keys())
    print(f"Found {len(common_keys)} common files")
    
    ssims, psnrs = [], []
    for key in common_keys:
        r_im = Image.open(gtr_dict[key]).resize(im_res)
        g_im = Image.open(gen_dict[key]).resize(im_res)
        # 计算 SSIM
        ssim = getSSIM(np.array(r_im), np.array(g_im))
        ssims.append(ssim)
        # 计算 PSNR
        r_im = r_im.convert("L")
        g_im = g_im.convert("L")
        psnr = getPSNR(np.array(r_im), np.array(g_im))
        psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="PyTorch FUnIE-GAN SSIM and PSNR Metric Runner")
    parser.add_argument("--image-data", default=r"E:\Underwater_Image_Enhancement\UIEBPairTest\waternet\UIEB_pair", type=str, metavar="PATH",
                        help="path to images ")
    parser.add_argument("--label-data", default=r"E:\Underwater_Image_Enhancement\demo01_ws\demo01_ws\CycleCAGAN\datasets\UIEB\test_Gtr", type=str, metavar="PATH",
                        help="path to ground truths")

    args = parser.parse_args()

    # Compute SSIM and PSNR
    SSIM_measures, PSNR_measures = SSIMs_PSNRs(
        args.label_data, args.image_data)
    print("SSIM on {0} samples".format(len(SSIM_measures)))
    print(f"Mean={np.mean(SSIM_measures):.3f}, Std={np.std(SSIM_measures):.3f}")
    print("PSNR on {0} samples".format(len(PSNR_measures)))
    print(f"Mean={np.mean(PSNR_measures):.3f}, Std={np.std(PSNR_measures):.3f}")
# SSIM on 515 samples
# Mean=0.803, Std=0.073
# PSNR on 515 samples
# Mean=27.410, Std=2.752
#复现的FUNIE代码




# SSIM on 515 samples
# Mean=0.82, Std=0.054
# PSNR on 515 samples
# Mean=27.689, Std=3.601
#ouput

# SSIM on 515 samples
# Mean=0.822, Std=0.063
# PSNR on 515 samples
# Mean=28.346, Std=2.947


# SSIM on 506 samples
# Mean=0.821, Std=0.063
# PSNR on 506 samples
# Mean=28.333, Std=2.962
# 188


# 去CBAM消融
# SSIM on 506 samples
# Mean=0.804, Std=0.059
# PSNR on 506 samples
# Mean=26.746, Std=2.656

# 去SR模块消融
# SSIM on 506 samples
# Mean=0.772, Std=0.068
# PSNR on 506 samples
# Mean=26.468, Std=2.568

# 去ACA模块消融
# SSIM on 506 samples
# Mean=0.808, Std=0.063
# PSNR on 506 samples
# Mean=27.591, Std=2.774

# OUrs UIEBPAIR
# Found 8 common files
# SSIM on 8 samples
# Mean=0.870, Std=0.083
# PSNR on 8 samples
# Mean=22.567, Std=4.633

# puie UIEB PAIR
# Found 8 common files
# SSIM on 8 samples
# Mean=0.795, Std=0.063
# PSNR on 8 samples
# Mean=21.401, Std=3.054

# funie gan uieb
# Found 8 common files
# SSIM on 8 samples
# Mean=0.695, Std=0.089
# PSNR on 8 samples
# Mean=20.839, Std=3.652

# rghs UIEB Pair
# Found 8 common files
# SSIM on 8 samples
# Mean=0.794, Std=0.075
# PSNR on 8 samples
# Mean=20.054, Std=2.690

# ruenet uieb pair 
# Found 8 common files
# SSIM on 8 samples
# Mean=0.820, Std=0.069
# PSNR on 8 samples
# Mean=21.132, Std=2.298

# shallowuwnet uieb pair 
# Found 8 common files
# SSIM on 8 samples
# Mean=0.733, Std=0.082
# PSNR on 8 samples
# Mean=21.651, Std=4.033

# ULAP UIEB Pair 
# Found 8 common files
# SSIM on 8 samples
# Mean=0.788, Std=0.046
# PSNR on 8 samples
# Mean=18.744, Std=2.157

# UWCYCLEGAN UIEB Pair 
# Found 8 common files
# SSIM on 8 samples
# Mean=0.747, Std=0.038
# PSNR on 8 samples
# Mean=19.668, Std=1.600

# waternet uieb pair 
# Found 8 common files
# SSIM on 8 samples
# Mean=0.836, Std=0.094
# PSNR on 8 samples
# Mean=21.983, Std=3.357