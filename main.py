import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import *
import seaborn as sns
from random import random
matrix = []
PATH_TO_DATA  = "./data/NIST/"
PATH_TO_IMAGE = PATH_TO_DATA + "figs_0/f0164_05.png"

img = cv.imread(PATH_TO_IMAGE, cv.IMREAD_GRAYSCALE)
sigma = 2 # experimental solution


img = cv.GaussianBlur(img, (25,25), sigmaX = sigma,sigmaY= sigma)
img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
          cv.THRESH_BINARY, 11, 2)
# plt.imshow(img, 'gray')
# plt.title("sigma = " + str(sigma))

#plt.show()


def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    size = np.size(img)
    while (True):
        dilated = cv.dilate(img, element)
        temp = cv.erode(dilated, element)
        temp = cv.subtract(temp,img)
        skel = cv.bitwise_or(skel, temp)
        img = dilated.copy()

        zeros =  cv.countNonZero(img)
        if zeros == size:
            break
    #return cv.bitwise_not(skel)
    return skel

img = skeletonize(img)
# plt.imshow(img, 'gray')
# plt.title("huy")
# plt.show()

def gerate_k(size:int, angle, sigmad = 0.07, sigmal = 1.):
    if size % 2 != 1:
        raise RuntimeError("size % 2 != 1")
    xc, yc = size/2, size/2
    res = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            d = sin((atan2(x - xc, y - yc) - angle)/2)
            l2 = ((x - xc)*(x - xc) + (y - yc)*(y - yc))
            res[x][y] =  exp(-(d/sigmad)**2 - l2/sigmal**2)
    return res

def get_pics(hist):
    """sorted list of pares value - coordinate"""
    size = len(hist)
    list = []
    for x in range(size):
        if hist[(x - 1) % size] < hist[x] >= hist[(x+ 1) % size]:
            list.append((hist[x], x))

    list.sort(reverse=True)
    return list

size = 17
num = 32

k = np.array([
    gerate_k(size, i * 2 * pi / num, sigmal = size/2)
    for i in range(num)
])


for i in range(len(k)):
    plt.subplot(4, 8, i + 1)
    plt.imshow(k[i]*255, 'gray')
    plt.title("huy" + str(i))
plt.show()
img = img/1.
filters = [cv.filter2D(img, -1, k[i]) for i in range(num)]
print(img.shape, filters[0].shape)

# for i in range(len(k)):
#     plt.subplot(4, 8, i+1)
#     plt.imshow(filters[i], cmap='gray')
#     plt.title("huy" + str(i))
# plt.show()
#
res = np.zeros((img.shape[0], img.shape[1], 3))
point = 0
colors = ['r', 'g', 'b']
pix = np.array([[img.max(), 0 , 0 ],[0, img.max(), 0],[0 , 0, img.max()]])

THRESHOLD = 0.6

for x in range(img.shape[0]):
    for y in range(img.shape[0]):
        if(img[x][y] > 0):
            v = img[x][y]
            res[x][y] = np.array([v, v, v])
            hist = np.array([filters[i][x][y] for i in range(num)])
            pics = get_pics(hist)
            num_pics = 0
            for val0, ang0 in pics:
                if val0 < THRESHOLD*pics[0][0]:
                    break
                num_pics += 1
            if num_pics == 1:
                for j in range(-2, 3):
                     for i in range(-2, 3):
                         if (0 < x + i < img.shape[0]) and (0 < y + j < img.shape[0]):
                            res[x + i][y + j] = pix[2]
            if num_pics == 3:
                for j in range(-2, 3):
                     for i in range(-2, 3):
                         if (0 < x + i < img.shape[0]) and (0 < y + j < img.shape[0]):
                            res[x + i][y + j] = pix[1]
                if (random() < 1. / 50) and point < 1:
                    plt.subplot(1, 2, 1)
                    plt.plot(hist, color=colors[1])
                    # for j in range(-3, 4):
                    #     for i in range(-3, 4):
                    #         res[x + i][y + j] = pix[point]
                    point += 1
                    print(pics)
            if num_pics > 3:
                for j in range(-2, 3):
                     for i in range(-2, 3):
                         if (0 < x + i < img.shape[0]) and (0 < y + j < img.shape[0]):
                            res[x + i][y + j] = pix[0]
plt.subplot(1, 2, 2)
plt.imshow(res/res.max())
plt.title("img")
plt.show()
#
# for i in range(len(k)):
#     plt.subplot(4, 8, i + 1)
#     sns.distplot(filters[i].reshape(-1))
#     plt.title("dist" + str(i))
# plt.show()