import cv2
import math
import numpy as np
 
def uicm(img):
    b, r, g = cv2.split(img)
    RG = r - g
    YB = (r + g)/2 - b
    m, n, o = np.shape(img)  #img为三维 rbg为二维
    K = m*n
    alpha_L = 0.1
    alpha_R = 0.1  ##参数α 可调
    T_alpha_L = math.ceil(alpha_L*K)  #向上取整
    T_alpha_R = math.floor(alpha_R*K) #向下取整
 
    RG_list = RG.flatten()
    RG_list = sorted(RG_list)
    sum_RG = 0
    for i in range(T_alpha_L+1, K-T_alpha_R ):
        sum_RG = sum_RG + RG_list[i]
    U_RG = sum_RG/(K - T_alpha_R - T_alpha_L)
    squ_RG = 0
    for i in range(K):
        squ_RG = squ_RG + np.square(RG_list[i] - U_RG)
    sigma2_RG = squ_RG/K
 
    YB_list = YB.flatten()
    YB_list = sorted(YB_list)
    sum_YB = 0
    for i in range(T_alpha_L+1, K-T_alpha_R ):
        sum_YB = sum_YB + YB_list[i]
    U_YB = sum_YB/(K - T_alpha_R - T_alpha_L)
    squ_YB = 0
    for i in range(K):
        squ_YB = squ_YB + np.square(YB_list[i] - U_YB)
    sigma2_YB = squ_YB/K
 
    Uicm = -0.0268*np.sqrt(np.square(U_RG) + np.square(U_YB)) + 0.1586*np.sqrt(sigma2_RG + sigma2_YB)
    return Uicm
 
def EME(rbg, L):
    m, n = np.shape(rbg)  #横向为n列 纵向为m行
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    #A1 = np.zeros((L, L))
    m1 = 0
    E = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1+L, n1:n1+L]
            rbg_min = np.amin(np.amin(A1))
            rbg_max = np.amax(np.amax(A1))
 
            if rbg_min > 0 :
                rbg_ratio = rbg_max/rbg_min
            else :
                rbg_ratio = rbg_max  ###
            E = E + np.log(rbg_ratio + 1e-5)
 
            n1 = n1 + L
        m1 = m1 + L
    E_sum = 2*E/(number_m*number_n)
    return E_sum
 
def UICONM(rbg, L):  #wrong
    m, n, o = np.shape(rbg)  #横向为n列 纵向为m行
    number_m = math.floor(m/L)
    number_n = math.floor(n/L)
    A1 = np.zeros((L, L)) #全0矩阵
    m1 = 0
    logAMEE = 0
    for i in range(number_m):
        n1 = 0
        for t in range(number_n):
            A1 = rbg[m1:m1+L, n1:n1+L]
            rbg_min = int(np.amin(np.amin(A1)))
            rbg_max = int(np.amax(np.amax(A1)))
            plip_add = rbg_max+rbg_min-rbg_max*rbg_min/1026
            if 1026-rbg_min > 0:
                plip_del = 1026*(rbg_max-rbg_min)/(1026-rbg_min)
                if plip_del > 0 and plip_add > 0:
                    local_a = plip_del/plip_add
                    local_b = math.log(plip_del/plip_add)
                    phi = local_a * local_b
                    logAMEE = logAMEE + phi
            n1 = n1 + L
        m1 = m1 + L
    logAMEE = 1026-1026*((1-logAMEE/1026)**(1/(number_n*number_m)))
    return logAMEE
 
if __name__ == '__main__':
    img = cv2.imread('/home/sandwich/demo01_ws/FUnIE-GAN-PyTorch-master/FUnIE-GAN-PyTorch-master/output_images/test_p2_.jpg')
    r, b, g = cv2.split(img)
 
    Uicm = uicm(img)
 
    EME_r = EME(r, 8)
    EME_b = EME(b, 8)
    EME_g = EME(g, 8)
    Uism = 0.299*EME_r + 0.144*EME_b + 0.557*EME_g
 
    Uiconm = UICONM(img, 8)
 
    uiqm = 0.0282*Uicm + 0.2953*Uism + 3.5753*Uiconm
    print(uiqm)