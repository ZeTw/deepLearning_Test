import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
实现图像与卷积核的卷积运算
高斯模糊之类的运算 肯定要求输出
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
    plt.imshow(img_result)
    plt.imsave('test.png', img_result)
    # print((img_array == img_result).all())

    plt.show()


img_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
img_filter = 1 / 16 * img_filter

img = cv2.imread('test.png')
print(img.shape)
exit()
# plt.imshow(img)
# plt.show()
# img = np.arange(25).reshape(5, 5)
# print(img)

# img_GaussianBlur = cv2.GaussianBlur(img, (3, 3), 0)
# cv2.imshow('img_g', img_GaussianBlur)
# cv2.waitKey(0)
conv2(img, img_filter)
# imgresult = img_filter * img[:, 0:3, 0:3]
# print(imgresult)
