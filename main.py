from math import *
from random import random

import cv2 as cv
import numpy as np
import seaborn as sns
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize, medial_axis

matrix = []
PATH_TO_DATA  = "./data/NIST/"
PATH_TO_IMAGE = PATH_TO_DATA + "figs_0/f0017_06.png"
#PATH_TO_IMAGE = PATH_TO_DATA + "figs_0/f0023_04.png"

img0 = cv.imread(PATH_TO_IMAGE, cv.IMREAD_GRAYSCALE)
sigma = 2 # experimental solution


img = cv.GaussianBlur(img0, (25,25), sigmaX = sigma,sigmaY= sigma)
img0 =  1 - cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY, 17, 2)

img = skeletonize(img0).astype(int)

# plt.imshow(img, 'gray')
# plt.title("sigma = " + str(sigma))

#plt.show()

#
# def skeletonize(img):
#     skel = np.zeros(img.shape, np.uint8)
#     element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
#     size = np.size(img)
#     while (True):
#         dilated = cv.dilate(img, element)
#         temp = cv.erode(dilated, element)
#         temp = cv.subtract(temp,img)
#         skel = cv.bitwise_or(skel, temp)
#         img = dilated.copy()
#
#         zeros =  cv.countNonZero(img)
#         if zeros == size:
#             break
#     #return cv.bitwise_not(skel)
#     return skel



def mod_dfs(imag, mask, x, y, PIX_SIZE_HALF):
    mask[x][y] = 1
    for i in range(-1, 2):
        if 0 <= x + i <= PIX_SIZE_HALF * 2:
           for j in range(-1, 2):
               if 0 <= y + j <= PIX_SIZE_HALF * 2:
                   if imag[x + i][y + j] and not mask[x + i][y + j]:
                       mod_dfs(imag, mask, x + i, y + j, PIX_SIZE_HALF)


def get_num(imag, x, y, PIX_SIZE_HALF):
    mat = imag[x - PIX_SIZE_HALF: x + PIX_SIZE_HALF + 1, y - PIX_SIZE_HALF: y + PIX_SIZE_HALF + 1]
    mask = np.zeros((PIX_SIZE_HALF * 2 + 1, PIX_SIZE_HALF * 2 + 1))
    mod_dfs(mat, mask, PIX_SIZE_HALF, PIX_SIZE_HALF, PIX_SIZE_HALF)
    res = 0
    for i in range(1, PIX_SIZE_HALF * 2 + 1):
        if mask[0][i] - mask[0][i - 1] == -1:
            res += 1
        if mask[i][PIX_SIZE_HALF * 2] - mask[i - 1][PIX_SIZE_HALF * 2] == -1:
            res += 1
        if mask[PIX_SIZE_HALF * 2][PIX_SIZE_HALF * 2  - i] - mask[PIX_SIZE_HALF * 2][PIX_SIZE_HALF * 2 + 1 - i] == -1:
            res += 1
        if mask[PIX_SIZE_HALF * 2 - i][0] - mask[PIX_SIZE_HALF * 2 + 1 - i][0] == -1:
            res += 1
    return res

res3 = np.zeros((img.shape[0], img.shape[1], 3))

def mark_points(img, PIX_SIZE_HALF = 12):
    for x in range(PIX_SIZE_HALF, img.shape[0] - PIX_SIZE_HALF):
        for y in range(PIX_SIZE_HALF, img.shape[1] - PIX_SIZE_HALF):
            if(img[x][y] > 0):
                r = get_num(img, x, y, PIX_SIZE_HALF)
                if r == 1:
                    res3[x][y] = np.array([0, 0, 1.])
                elif r == 2:
                    res3[x][y] = np.array([1., 1., 1.])
                elif r == 3:
                    res3[x][y] = np.array([0, 1., 0])
                elif r > 3:
                    res3[x][y] = np.array([1., 0, 0])


mark_points(img, 2)

plt.subplot(1, 2, 1)
plt.imshow(img0, 'gray')
plt.title("img_tr")

plt.subplot(1, 2, 2)
plt.imshow(res3/res3.max())
plt.title("img_points")
plt.show()


# plt.subplot(1, 2, 1)
# plt.imshow(img0, 'gray')
# plt.title("img_tr")
#
#
# coords = corner_peaks(corner_harris(img), min_distance=3)
# coords_subpix = corner_subpix(img, coords, window_size=10)
#
#
# plt.subplot(1, 2, 2)
# plt.imshow(img, 'gray')
# plt.title("img_skel")
#
# plt.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=4)
# plt.show()


# # plt.imshow(img, 'gray')
# # plt.title("huy")
# # plt.show()
#
# def gerate_k(size:int, angle, sigmad = 0.2, sigmal = 1.):
#     if size % 2 != 1:
#         raise RuntimeError("size % 2 != 1")
#     xc, yc = size/2, size/2
#     res = np.zeros((size, size))
#     for x in range(size):
#         for y in range(size):
#             d = sin((atan2(x - xc, y - yc) - angle)/2)
#             l2 = 1 /(((x - xc)*(x - xc) + (y - yc)*(y - yc)) + 0.001)
#             res[x][y] =  exp(-(d/sigmad)**2 - l2/sigmal**2)
#     return res
#
# def get_pics(hist):
#     """sorted list of pares value - coordinate"""
#     size = len(hist)
#     list = []
#     for x in range(size):
#         if hist[(x - 1) % size] < hist[x] >= hist[(x+ 1) % size]:
#             list.append((hist[x], x))
#
#     list.sort(reverse=True)
#     return list
#
# size = 13
# num = 32
#
# k = np.array([
#     gerate_k(size, i * 2 * pi / num, sigmal=size/2)
#     for i in range(num)
# ])
#
#
# for i in range(len(k)):
#     plt.subplot(4, 8, i + 1)
#     plt.imshow(k[i]*255, 'gray')
#     plt.title("huy" + str(i))
# plt.show()
# img = img/1.
# filters = [cv.filter2D(img, -1, k[i]) for i in range(num)]
# print(img.shape, filters[0].shape)
#
# # for i in range(len(k)):
# #     plt.subplot(4, 8, i+1)
# #     plt.imshow(filters[i], cmap='gray')
# #     plt.title("huy" + str(i))
# # plt.show()
# #
# point = 0
# colors = ['r', 'g', 'b']
# pix = np.array([[img.max(), 0 , 0 ],[0, img.max(), 0],[0 , 0, img.max()]])
#
# THRESHOLD = 0.6
#
#
# def get_4channels(hist):
#     sum = hist.sum()/hist.max()
#     hist = sorted( enumerate( hist.copy()/hist.max()), key=lambda x: x[1], reverse=True)
#
#     n = len(hist)
#     k = [n / 4, n / 12, n / 12]
#
#     i = [0]
#
#     for ind , i_val in enumerate(hist[1:]):
#         c = len(i) - 1
#         if c == 3:
#             break
#         ok = True
#         for j in i:
#             #print('#',hist[j][0], i_val[0], abs(hist[j][0] - i_val[0]), k[c], abs(hist[j][0] - i_val[0]) <= k[c])
#             if abs(hist[j][0] - i_val[0]) <= k[c]:
#                 ok = False
#                 break
#         #print('###', ok)
#         if ok:
#             i.append(ind+ 1)
#         #print(i)
#     #print([hist[j] for j in i])
#     eps = 0.0
#
#     val = np.array([hist[j][1] for j in i])
#     res = np.array([val[0:j].sum() * ((1./j + eps) + j/(n - j)) - sum/(n - j) for j in range(1, 5)])
#     return res/res.sum()
#
#
#
#
#
#
#     return np.array([ch1, ch2, 0, 0])
#
# # for x in range(img.shape[0]):
# #     for y in range(img.shape[0]):
# #         if(img[x][y] > 0):
# #             v = img[x][y]
# #             res[x][y] = np.array([v, v, v])
# #             hist = np.array([filters[i][x][y] for i in range(num)])
# #             pics = get_pics(hist)
# #             num_pics = 0
# #             for val0, ang0 in pics:
# #                 if val0 < THRESHOLD*pics[0][0]:
# #                     break
# #                 num_pics += 1
# #             if num_pics == 1:
# #                 for j in range(-2, 3):
# #                      for i in range(-2, 3):
# #                          if (0 < x + i < img.shape[0]) and (0 < y + j < img.shape[0]):
# #                             res[x + i][y + j] = pix[2]
# #             if num_pics == 3:
# #                 for j in range(-2, 3):
# #                      for i in range(-2, 3):
# #                          if (0 < x + i < img.shape[0]) and (0 < y + j < img.shape[0]):
# #                             res[x + i][y + j] = pix[1]
# #                 if (random() < 1. / 50) and point < 1:
# #                     plt.subplot(1, 2, 1)
# #                     plt.plot(hist, color=colors[1])
# #                     # for j in range(-3, 4):
# #                     #     for i in range(-3, 4):
# #                     #         res[x + i][y + j] = pix[point]
# #                     point += 1
# #                     print(pics)
# #             if num_pics > 3:
# #                 for j in range(-2, 3):
# #                      for i in range(-2, 3):
# #                          if (0 < x + i < img.shape[0]) and (0 < y + j < img.shape[0]):
# #                             res[x + i][y + j] = pix[0]
#
#
# res3 = np.zeros((img.shape[0], img.shape[1], 3))
# res_l = np.zeros((img.shape[0], img.shape[1]))
# for x in range(img.shape[0]):
#     for y in range(img.shape[0]):
#         if(img[x][y] > 0):
#             hist = np.array([filters[i][x][y] for i in range(num)])
#             ch4 = get_4channels(hist)
#             ind = np.argmax(ch4)
#             if ind == 0:
#                 res3[x][y] = np.array([1., 0, 0])
#             if ind == 2:
#                 res3[x][y] = np.array([0, 1., 0])
#             if ind == 3:
#                 res3[x][y] = np.array([0, 0, 1.])
#             res_l[x][y] = 1. if ch4.max() == ch4[1] else 0
#
# plt.subplot(1, 2, 1)
# plt.imshow(res_l, 'gray')
# plt.title("img_lines")
#
# plt.subplot(1, 2, 2)
# plt.imshow(res3/res3.max())
# plt.title("img_points")
# plt.show()
#
# for i in range(len(k)):
#     plt.subplot(4, 8, i + 1)
#     sns.distplot(filters[i].reshape(-1))
#     plt.title("dist" + str(i))
# plt.show()