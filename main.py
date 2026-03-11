import numpy as np #bài tập lớn, đề 6: Hệ thống phát hiện biên ảnh trong điều kiện nhiễu
import cv2
import math
import matplotlib.pyplot as plt
import sys
import os   #1; ham loc nhieu, 2; ham phat hien bien voi 3 bo loc
sobel_Hx=np.array([ #bộ lọc sobel #bt 19-1-2026
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]])
sobel_Hy=np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]])
laplace_H2=np.array([   #bộ lọc laplace #bt 26-1-2026
    [1,1,1],
    [1,-8,1],
    [1,1,1]])
laplace_H3=np.array([
    [-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]])

#lọc nhiễu #bt 17-12-2025
locnhieu_H3=np.array([  #khai báo bộ lọc H3
    [1,2,1],
    [2,4,2],
    [1,2,1]])/16
#su dung filter2D cua cv2 de tinh tich chap 2 chieu (ap dung H3 len anh)
def locnhieu(duongdan):
    img=cv2.imread(duongdan,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Lỗi")
        return None
    img_da_loc=cv2.filter2D(img,-1,locnhieu_H3) #ham de ap hat nhan (kernal) len anh, thuc hien tich chap 2 chieu; (ảnh vào, độ sâu bit đầu ra, bộ lọc (kernal))
    return img_da_loc

def bienanh_sobel(img):
    pad_width=1 # Zero paddin (đệm vùng biên bằng 0)
    img_pad = np.pad(img, pad_width, mode='constant', constant_values=0)
    Gx, Gy, img_gradient = [np.zeros_like(img_pad, dtype=np.int32) for _ in range(3)]
    height, width=img_pad.shape #cao, rong
    for h in range (1, height-1):
        for w in range (1, width-1):
            window=img_pad[h-1:h+2, w-1:w+2] #lay ra 1 cua so roi tinh #gon hon bien duoi
            Gx[h, w] = np.sum(window * sobel_Hx)
            Gy[h, w] = np.sum(window * sobel_Hy)
            g= math.sqrt(Gx[h, w]**2 + Gy[h, w]**2)  #G=căn bậc 2(Gx^2​+Gy^2)​
            img_gradient[h, w] = min(255, int(g))
    img_gradient = img_gradient[1:-1, 1:-1]
    return img_gradient.astype(np.uint8)
def bienanh_laplace(img):
    pad_width=1 # Zero paddin (đệm vùng biên bằng 0)
    img_pad = np.pad(img, pad_width, mode='constant', constant_values=0)
    I_h2, I_h3 = [np.zeros_like(img_pad, dtype=np.int32) for _ in range(2)]
    height, width=img_pad.shape #cao, rong
    for h in range (1, height-1):
        for w in range (1, width-1):
            window=img_pad[h-1:h+2, w-1:w+2] #lay ra 1 cua so roi tinh #gon hon bien duoi
            I_h2[h, w] = np.sum(window * laplace_H2)
            I_h3[h, w] = np.sum(window * laplace_H3)
    I_h2=I_h2[1:-1, 1:-1]
    I_h3=I_h3[1:-1, 1:-1]
    # Với H2 (Tìm biên): Lấy trị tuyệt đối để biên luôn sáng, sau đó clip
    I_h2_final = np.clip(np.abs(I_h2), 0, 255).astype(np.uint8)
    # Với H3 (Làm nét): Thường giá trị đã dương, nhưng vẫn cần clip để tránh vượt quá 255
    I_h3_final = np.clip(I_h3, 0, 255).astype(np.uint8)
    return I_h2_final.astype(np.uint8), I_h3_final.astype(np.uint8)
def bienanh_canny(img):
    img_canny = cv2.Canny(img, 100, 200)
    return img_canny

def main():
    duongdan="D:/python_code/#xulyanh/btl_k2y4/anh_dau_vao/anhgoc(1).jpg"
    img_goc=cv2.imread(duongdan,cv2.IMREAD_COLOR)
    if img_goc is None:
        print("Lỗi")
        return None
    img=locnhieu(duongdan)
    img_kq_sobel=bienanh_sobel(img)
    img_kq_laplace1, img_kq_laplace2=bienanh_laplace(img)
    img_kq_canny=bienanh_canny(img)
    cv2.imshow("anh goc",img_goc)
    cv2.imshow("anh phat hien bien (sobel)",img_kq_sobel)
    cv2.imshow("anh phat hien bien (laplace) (H2)",img_kq_laplace1)
    cv2.imshow("anh phat hien bien (laplace) (H3)",img_kq_laplace2)
    cv2.imshow("anh phat hien bien (canny)",img_kq_canny)
    
    # Hiển thị bằng Matplotlib
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.title("Anh goc")
    plt.imshow(cv2.cvtColor(img_goc, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(2,3,2)
    plt.title("Sau loc nhieu")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(2,3,3)
    plt.title("Sobel")
    plt.imshow(img_kq_sobel, cmap="gray")
    plt.axis("off")
    plt.subplot(2,3,4)
    plt.title("Laplace H2")
    plt.imshow(img_kq_laplace1, cmap="gray")
    plt.axis("off")
    plt.subplot(2,3,5)
    plt.title("Laplace H3")
    plt.imshow(img_kq_laplace2, cmap="gray")
    plt.axis("off")
    plt.subplot(2,3,6)
    plt.title("Canny")
    plt.imshow(img_kq_canny, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('D:/python_code/#xulyanh/btl_k2y4/anh_ket_qua/anh_da_loc1.jpg', 5) #ham de xuat anh ra
if __name__ == "__main__":
    main()