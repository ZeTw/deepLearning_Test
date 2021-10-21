import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
equalization(img_array): 
    https://blog.csdn.net/schwein_van/article/details/84336633
    直方图均衡化:对灰度图像进行处理 , 彩色图像的均衡化可以对每个通道单独处理之后再将通道进行拼接
    img_array:单通道灰度图数组
    num_array:单通道灰度图采用8位表示像素，灰度范围 0-255 该数组用来保存图像中每个灰度的数目
    N:该图像包含的像素个数
    pr: 用于记录所有灰度占整个图像的比例
    pix_min:图像中的最小灰度值 pin_max:图像中的最大灰度值
    s:用于存储经过处理之后的新的灰度值
    最后将灰度值按照对应关系更新原图像数组

conv2(img_array, img_filter, axis=1):
    手写卷积，卷积核不同可实现边缘检测或者去噪 单通道、多通道都可以
    目前输入和输出的大小是完全一致的，且stride=1
'''


def conv2(img_array, img_filter, axis=1):
    img_ndim = img_array.ndim
    img_result = np.zeros_like(img_array)
    filter_w, filter_h = img_filter.shape
    padding = (filter_w - 1) // 2
    if img_ndim == 2:
        img_h, img_w = img_array.shape
        img_array2 = np.pad(img_array, ((padding, padding), (padding, padding)))
        for h in range(img_h):
            for w in range(img_w):
                img_result[h][w] = (img_filter * img_array2[h:h + filter_h, w:w + filter_w]).sum()
    else:
        img_h, img_w, img_c = img_array.shape
        img_array2 = np.pad(img_array, ((padding, padding), (padding, padding), (0, 0)))
        for c in range(img_c):
            for h in range(img_h):
                for w in range(img_w):
                    img_result[h][w][c] = (img_filter * img_array2[h:h + filter_h, w:w + filter_w, c]).sum()
    # 需要注意的是，这里使用plt来展示和保存图片的话，会出现颜色变化，改为opencv就可以正常展示和保存了
    cv2.imwrite('test.jpg', img_result)
#     plt.imshow(img_result)
#     plt.imsave('test.png', img_result)
  
#     plt.show()


img_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
img_filter = 1 / 16 * img_filter

img = cv2.imread('test.png')

conv2(img, img_filter)

img = cv2.imread('0043.jpg')
# equalization(img)
img_result = np.zeros_like(img, dtype=int)
for i in range(img.shape[2]):
    img_result[..., i] = equalization(img[..., i])

cv2.imwrite('3_test.jpg', img_result)
