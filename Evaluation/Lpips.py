import os
import torch
import lpips
from PIL import Image
import numpy as np
from tqdm import tqdm

# ========== 超参数定义 ==========
FOLDER_GT = r"E:\Underwater_Image_Enhancement\demo01_ws\demo01_ws\CycleCAGAN\datasets\UIEB\test_Gtr"     # Ground Truth 图像文件夹
FOLDER_ENH = r"E:\Underwater_Image_Enhancement\UIEBPairTest\UIEB_PAIR_SHALLOWUWNET"        # 增强后图像文件夹
NET_TYPE = "alex"                       # 可选: 'alex', 'vgg', 'squeeze'

# ========== LPIPS 计算函数 ==========
def load_image_as_tensor(path, device):
    """加载图像并转为符合LPIPS输入要求的Tensor"""
    img = Image.open(path).convert("RGB")
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor.to(device)

def calculate_lpips_for_folders(folder_gt, folder_enh, net_type="alex"):
    """
    比较两个文件夹中的图像并计算平均LPIPS距离
    A: Ground Truth
    B: Enhanced
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 当前计算设备: {device}")

    # 加载 LPIPS 模型
    lpips_model = lpips.LPIPS(net=net_type).to(device)
    lpips_model.eval()

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    files_gt = sorted([f for f in os.listdir(folder_gt) if f.lower().endswith(valid_exts)])
    files_enh = sorted([f for f in os.listdir(folder_enh) if f.lower().endswith(valid_exts)])
    common_files = sorted(list(set(files_gt) & set(files_enh)))

    if not common_files:
        print("❌ 两个文件夹中没有同名图片！")
        return

    distances = []
    for fname in tqdm(common_files, desc="Comparing images"):
        path_gt = os.path.join(folder_gt, fname)
        path_enh = os.path.join(folder_enh, fname)
        try:
            img_gt = load_image_as_tensor(path_gt, device)
            img_enh = load_image_as_tensor(path_enh, device)
            with torch.no_grad():
                dist = lpips_model(img_gt, img_enh).item()  # 注意顺序: GT, Enhanced
            distances.append(dist)
        except Exception as e:
            print(f"⚠️ 跳过 {fname}: {e}")

    if distances:
        mean_lpips = np.mean(distances)
        std_lpips = np.std(distances)
        print("\n===== 📊 LPIPS 统计结果 =====")
        print(f"平均 LPIPS 距离: {mean_lpips:.6f}")
        print(f"标准差: {std_lpips:.6f}")
    else:
        print("未成功计算任何图片的 LPIPS 距离。")

# ========== 主执行入口 ==========
if __name__ == "__main__":
    calculate_lpips_for_folders(FOLDER_GT, FOLDER_ENH, NET_TYPE)






# Ours EUVP TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.117141
# 标准差: 0.064520

# RUENET EUVP Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.125561
# 标准差: 0.073020

# SHallowUWNet EUVP Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.230119
# 标准差: 0.076859

# UW-CycleGan EUVP test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.248767
# 标准差: 0.061720

# Ulap EUVP TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.238495
# 标准差: 0.069141

# RGHS EUVP TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.238536
# 标准差: 0.069376

# puie EUVP Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.239026
# 标准差: 0.067316

# waternet EUVP Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.219444
# 标准差: 0.068291

#  FUnieGAN EUVP Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.222262
# 标准差: 0.078233








# Ours UFO TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.180987
# 标准差: 0.050388

# RUE-Net UFO Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.173799
# 标准差: 0.050298

# PUIE UFO Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.307064
# 标准差: 0.078259

# UWCycleGAN UFO Test 
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.297007
# 标准差: 0.069926

# waternet UFO TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.286273
# 标准差: 0.079597

# ulap ufo test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.305380
# 标准差: 0.080262

# RGHS UFO TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.303719
# 标准差: 0.079795

# shallowuwnet ufo test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.294628
# 标准差: 0.086697

# FUNIE GAN UFO TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.282437
# 标准差: 0.085228


# Ours UIEB  pair Test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.072600
# 标准差: 0.041210

# Funie gan UIEB Pair 
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.229776
# 标准差: 0.06979

# PUIE UIEB Pair test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.114083
# 标准差: 0.023109

# UWCYCLEGAN PAIR TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.161543
# 标准差: 0.037184

# RGHS UIEB PAIR TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.072783
# 标准差: 0.037524

# ruenet uieb pair test
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.266578
# 标准差: 0.058551

# ULAP UIEB PAIR TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.209933
# 标准差: 0.052152

# WATERNET UIEB PAIR TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.086314
# 标准差: 0.033688

# SHALLOWUWNET UIEB PAIR TEST
# ===== 📊 LPIPS 统计结果 =====
# 平均 LPIPS 距离: 0.255993
# 标准差: 0.076086






