import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def process_image(image_path):
    """处理单张图片并返回UCIQE值及其组成部分"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return None
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    
    # 计算色度标准差
    delta = np.std(H) / 180
    
    # 计算饱和度平均值
    mu = np.mean(S) / 255
    
    # 计算亮度对比值
    n, m = np.shape(V)
    number = max(1, np.floor(n * m / 100).astype(int))  # 确保至少1个像素
    v = V.flatten() / 255
    v.sort()
    bottom = np.sum(v[:number]) / number
    top = np.sum(v[-number:]) / number
    conl = top - bottom
    
    # 计算UCIQE
    uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    
    return uciqe, delta, conl, mu

def process_folder(folder_path):
    """处理文件夹中的所有图片并返回结果统计"""
    uciqe_values = []  # 存储所有UCIQE值
    image_results = []  # 存储每张图片的结果
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 处理图片
            result = process_image(file_path)
            if result is not None:
                uciqe, delta, conl, mu = result
                uciqe_values.append(uciqe)
                image_results.append({
                    'filename': filename,
                    'uciqe': uciqe,
                    'delta': delta,
                    'conl': conl,
                    'mu': mu
                })
                print(f"处理完成: {filename} - UCIQE: {uciqe:.4f}")
    
    return uciqe_values, image_results

def generate_report(uciqe_values, image_results, folder_path):
    """生成分析报告并保存到文件"""
    # 计算基本统计量
    mean_uciqe = np.mean(uciqe_values)
    std_uciqe = np.std(uciqe_values)
    min_uciqe = np.min(uciqe_values)
    max_uciqe = np.max(uciqe_values)
    
    # 找出UCIQE最高的3张图片
    sorted_results = sorted(image_results, key=lambda x: x['uciqe'], reverse=True)
    top_images = sorted_results[:3]
    
    # 创建报告文件路径
    report_path = os.path.join(folder_path, "UCIQE_Report.txt")
    
    # 写入报告
    with open(report_path, 'w') as report_file:
        report_file.write("=" * 60 + "\n")
        report_file.write("图像质量分析报告 (UCIQE)\n")
        report_file.write("=" * 60 + "\n\n")
        
        report_file.write(f"分析文件夹: {folder_path}\n")
        report_file.write(f"图片总数: {len(uciqe_values)}\n\n")
        
        report_file.write("总体统计:\n")
        report_file.write(f"  UCIQE平均值: {mean_uciqe:.4f}\n")
        report_file.write(f"  UCIQE标准差: {std_uciqe:.4f}\n")
        report_file.write(f"  最低UCIQE值: {min_uciqe:.4f}\n")
        report_file.write(f"  最高UCIQE值: {max_uciqe:.4f}\n\n")
        
        report_file.write("UCIQE最高的3张图片:\n")
        for i, img in enumerate(top_images, 1):
            report_file.write(f"{i}. {img['filename']} - UCIQE: {img['uciqe']:.4f}\n")
            report_file.write(f"   - 色度标准差: {img['delta']:.4f}\n")
            report_file.write(f"   - 亮度对比值: {img['conl']:.4f}\n")
            report_file.write(f"   - 饱和度均值: {img['mu']:.4f}\n")
        
        report_file.write("\n详细结果:\n")
        report_file.write("文件名\t\tUCIQE\t色度标准差\t亮度对比值\t饱和度均值\n")
        for img in sorted(image_results, key=lambda x: x['filename']):
            report_file.write(f"{img['filename']}\t{img['uciqe']:.4f}\t{img['delta']:.4f}\t{img['conl']:.4f}\t{img['mu']:.4f}\n")
    
    print(f"分析报告已保存至: {report_path}")
    
    # 返回统计结果
    return {
        'mean_uciqe': mean_uciqe,
        'std_uciqe': std_uciqe,
        'top_images': top_images
    }

def visualize_results(uciqe_values, folder_path):
    """可视化UCIQE值分布"""
    plt.figure(figsize=(12, 6))
    
    # UCIQE值直方图
    plt.subplot(1, 2, 1)
    plt.hist(uciqe_values, bins=20, color='skyblue', edgecolor='black')
    plt.title('UCIQE值分布')
    plt.xlabel('UCIQE值')
    plt.ylabel('图片数量')
    
    # UCIQE值箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(uciqe_values, vert=False)
    plt.title('UCIQE值箱线图')
    plt.xlabel('UCIQE值')
    
    plt.tight_layout()
    plot_path = os.path.join(folder_path, "UCIQE_Distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"UCIQE分布图已保存至: {plot_path}")

if __name__ == "__main__":
    # 设置文件夹路径
    folder_path = r"E:\Underwater_Image_Enhancement\Our_Dataset\UWCycleGAN"  # 替换为您的文件夹路径
    
    # 处理文件夹中的所有图片
    uciqe_values, image_results = process_folder(folder_path)
    
    if not uciqe_values:
        print("未找到任何可处理的图片。请检查文件夹路径和图片格式。")
    else:
        # 生成报告
        stats = generate_report(uciqe_values, image_results, folder_path)
        
        # 可视化结果
        visualize_results(uciqe_values, folder_path)
        
        # 打印摘要
        print("\n" + "=" * 60)
        print(f"图片总数: {len(uciqe_values)}")
        print(f"UCIQE平均值: {stats['mean_uciqe']:.4f}")
        print(f"UCIQE标准差: {stats['std_uciqe']:.4f}")
        print("\nUCIQE最高的3张图片:")
        for i, img in enumerate(stats['top_images'], 1):
            print(f"{i}. {img['filename']} - UCIQE: {img['uciqe']:.4f}")
        print("=" * 60)




# EUVP TEST
# 图片总数: 515
# UCIQE平均值: 0.4220
# UCIQE标准差: 0.0669

# UCIQE最高的3张图片:
# 1. test_p461_.jpg - UCIQE: 0.6010
# 2. test_p384_.jpg - UCIQE: 0.5997
# 3. test_p478_.jpg - UCIQE: 0.5853 Puie

# 图片总数: 516
# UCIQE平均值: 0.4079
# UCIQE标准差: 0.0584

# UCIQE最高的3张图片:
# 1. test_p478_.png - UCIQE: 0.6003
# 2. test_p461_.png - UCIQE: 0.5788
# 3. test_p415_.png - UCIQE: 0.5766 FUnIE

# 图片总数: 516
# UCIQE平均值: 0.4195
# UCIQE标准差: 0.0606

# UCIQE最高的3张图片:
# 1. test_p461_.jpg - UCIQE: 0.5825
# 2. test_p384_.jpg - UCIQE: 0.5814
# 3. test_p415_.jpg - UCIQE: 0.5682 RUENET

# 图片总数: 515
# UCIQE平均值: 0.4091
# UCIQE标准差: 0.0673

# UCIQE最高的3张图片:
# 1. test_p478_.jpg - UCIQE: 0.6032
# 2. test_p461_.jpg - UCIQE: 0.5917
# 3. test_p384_.jpg - UCIQE: 0.5835 ShallowNet


# 图片总数: 516
# UCIQE平均值: 0.4386
# UCIQE标准差: 0.0433

# UCIQE最高的3张图片:
# 1. test_p461_.jpg - UCIQE: 0.5728
# 2. test_p384_.jpg - UCIQE: 0.5719
# 3. test_p478_.jpg - UCIQE: 0.5636 waternet


# 图片总数: 516
# UCIQE平均值: 0.4482
# UCIQE标准差: 0.0455

# UCIQE最高的3张图片:
# 1. test_p109_.jpg - UCIQE: 0.6044
# 2. test_p161_.jpg - UCIQE: 0.6018
# 3. test_p415_.jpg - UCIQE: 0.6013 Ours



# 图片总数：515
# UCIQE平均值: 0.3717
# UCIQE标准差: 0.0866

# UCIQE最高的3张图片:
# 1. test_p93_.jpg - UCIQE: 0.5485
# 2. test_p120_.jpg - UCIQE: 0.5322
# 3. test_p122_.jpg - UCIQE: 0.5305 Uwcycle-gan

# 图片总数: 515
# UCIQE平均值: 0.3815
# UCIQE标准差: 0.0420

# UCIQE最高的3张图片:
# 1. test_p461_.jpg - UCIQE: 0.6097
# 2. test_p478_.jpg - UCIQE: 0.6065
# 3. test_p384_.jpg - UCIQE: 0.6056 RGHS

# 图片总数: 515
# UCIQE平均值: 0.3892
# UCIQE标准差: 0.0663

# UCIQE最高的3张图片:
# 1. test_p478_.jpg - UCIQE: 0.6629
# 2. test_p244_.jpg - UCIQE: 0.6608
# 3. test_p336_.jpg - UCIQE: 0.6599 ULAP









# UFO120 TEST
# 图片总数: 120
# UCIQE平均值: 0.2920
# UCIQE标准差: 0.0396

# UCIQE最高的3张图片:
# 1. set_f2.jpg - UCIQE: 0.6035
# 2. set_u41.jpg - UCIQE: 0.6032
# 3. set_u36.jpg - UCIQE: 0.5870 RGHS

# 图片总数: 120
# UCIQE平均值: 0.4368
# UCIQE标准差: 0.0610

# UCIQE最高的3张图片:
# 1. set_u112.jpg - UCIQE: 0.5794
# 2. set_o1.jpg - UCIQE: 0.5785
# 3. set_o17.jpg - UCIQE: 0.5774 PUIE


# 图片总数: 120
# UCIQE平均值: 0.4333
# UCIQE标准差: 0.0593

# UCIQE最高的3张图片:
# 1. set_o17.jpg - UCIQE: 0.5789
# 2. set_u112.jpg - UCIQE: 0.5728
# 3. set_f47.jpg - UCIQE: 0.5564 RUENET


# 图片总数: 121
# UCIQE平均值: 0.4614
# UCIQE标准差: 0.0408

# UCIQE最高的3张图片:
# 1. set_u112.jpg - UCIQE: 0.5644
# 2. set_f47.jpg - UCIQE: 0.5506
# 3. set_f2.jpg - UCIQE: 0.5504 Ours  
#



# 图片总数: 120
# UCIQE平均值: 0.4260
# UCIQE标准差: 0.0666

# UCIQE最高的3张图片:
# 1. set_o17.jpg - UCIQE: 0.6138
# 2. set_f45.jpg - UCIQE: 0.5850
# 3. set_u112.jpg - UCIQE: 0.5632 ShallowNet 


# 图片总数: 120
# UCIQE平均值: 0.4158
# UCIQE标准差: 0.0601

# UCIQE最高的3张图片:
# 1. set_f45.jpg - UCIQE: 0.6643
# 2. set_o1.jpg - UCIQE: 0.6541
# 3. set_f2.jpg - UCIQE: 0.6421 UW CycleGAN

# 图片总数: 120
# UCIQE平均值: 0.4271
# UCIQE标准差: 0.0607

# UCIQE最高的3张图片:
# 1. set_u112.jpg - UCIQE: 0.5899
# 2. set_o17.jpg - UCIQE: 0.5893
# 3. set_f45.jpg - UCIQE: 0.5616 FUnIE-GAN


# 图片总数: 120
# UCIQE平均值: 0.4506
# UCIQE标准差: 0.0394

# UCIQE最高的3张图片:
# 1. set_f2.jpg - UCIQE: 0.5474
# 2. set_f47.jpg - UCIQE: 0.5356
# 3. set_u41.jpg - UCIQE: 0.5274 WaterNET


# 图片总数: 120
# UCIQE平均值: 0.3026
# UCIQE标准差: 0.0711

# UCIQE最高的3张图片:
# 1. set_o9.jpg - UCIQE: 0.6706
# 2. set_f2.jpg - UCIQE: 0.6580
# 3. set_o1.jpg - UCIQE: 0.6472 ULAP




# Unpair 

# UCIQE平均值: 0.3587
# UCIQE标准差: 0.0291

# ULAP

# UCIQE平均值: 0.3601
# UCIQE标准差: 0.0361

# RGHS

# UCIQE平均值: 0.4194
# UCIQE标准差: 0.0407

#  PUIE

# UCIQE平均值: 0.3964
# UCIQE标准差: 0.0483

#  ruenet

# UCIQE平均值: 0.4049
# UCIQE标准差: 0.0384

#  Shallownet


# UCIQE平均值: 0.3794
# UCIQE标准差: 0.0486

#   UW-CycleGAN

# UCIQE平均值: 0.4233
# UCIQE标准差: 0.0346

# waternet


# UCIQE平均值: 0.3857
# UCIQE标准差: 0.0463

# funie

# UCIQE平均值: 0.4338
# UCIQE标准差: 0.0489
# Ours





# 消融
# UCIQE平均值: 0.4050
# UCIQE标准差: 0.0623 无ACA

# UCIQE平均值: 0.4179
# UCIQE标准差: 0.0582 无CBAM

# UCIQE平均值: 0.3998
# UCIQE标准差: 0.0641 无SR


# OUrs UIEB PAIR TEST
# UCIQE平均值: 0.5785
# UCIQE标准差: 0.0378

# FUNIE GAN UIEB 
# UCIQE平均值: 0.4647
# UCIQE标准差: 0.0357

# PUIE UIEB
# UCIQE平均值: 0.4882
# UCIQE标准差: 0.0360

# rghs uieb
# UCIQE平均值: 0.4932
# UCIQE标准差: 0.0338

# RUENET UIEB
# UCIQE平均值: 0.5318
# UCIQE标准差: 0.0254

# SHALLOWUWNET UIEB
# UCIQE平均值: 0.5262
# UCIQE标准差: 0.0234

# ulap uieb
# UCIQE平均值: 0.4559
# UCIQE标准差: 0.0368

# UWCYCLEGAN UIEB
# UCIQE平均值: 0.5706
# UCIQE标准差: 0.0163

# WATERNET UIEB
# UCIQE平均值: 0.4707
# UCIQE标准差: 0.0335



# PUIE U45
# UCIQE平均值: 0.4211
# UCIQE标准差: 0.0288

# RGHS U45
# UCIQE平均值: 0.4582
# UCIQE标准差: 0.0266

# RUENET U45
# UCIQE平均值: 0.3640
# UCIQE标准差: 0.0684

# SHALLOW U45
# UCIQE平均值: 0.3658
# UCIQE标准差: 0.0620

# Ours U45
# UCIQE平均值: 0.4889
# UCIQE标准差: 0.0525

# FUNIE U45
# UCIQE平均值: 0.3080
# UCIQE标准差: 0.0621

# ulap u45
# UCIQE平均值: 0.2640
# UCIQE标准差: 0.0476

# UWCYCLEGAN U45
# UCIQE平均值: 0.4373
# UCIQE标准差: 0.0597

# waternet U45
# UCIQE平均值: 0.4197
# UCIQE标准差: 0.0270