import copy
import random

import cv2 as cv
import numpy as np
import png
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

matrix = []
PATH_TO_DATA = "./data/NIST/"
PATH_TO_IMAGE = PATH_TO_DATA + "figs_0/f0017_06.png"
# PATH_TO_IMAGE = PATH_TO_DATA + "figs_0/f0023_04.png"

img0 = cv.imread(PATH_TO_IMAGE, cv.IMREAD_GRAYSCALE)
sigma = 2  # experimental solution

img = cv.GaussianBlur(img0, (25, 25), sigmaX=sigma, sigmaY=sigma)
img0 = 1 - cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 17, 2)

img = skeletonize(img0).astype(int)


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
        if mask[PIX_SIZE_HALF * 2][PIX_SIZE_HALF * 2 - i] - mask[PIX_SIZE_HALF * 2][PIX_SIZE_HALF * 2 + 1 - i] == -1:
            res += 1
        if mask[PIX_SIZE_HALF * 2 - i][0] - mask[PIX_SIZE_HALF * 2 + 1 - i][0] == -1:
            res += 1
    return res


def mark_points(img, PIX_SIZE_HALF=12):
    for x in range(PIX_SIZE_HALF, img.shape[0] - PIX_SIZE_HALF):
        for y in range(PIX_SIZE_HALF, img.shape[1] - PIX_SIZE_HALF):
            if (img[x][y] > 0):
                r = get_num(img, x, y, PIX_SIZE_HALF)
                if r == 1:
                    img[x][y] = np.array([0, 0, 1.])
                elif r == 2:
                    img[x][y] = np.array([1., 1., 1.])
                elif r == 3:
                    img[x][y] = np.array([0, 1., 0])
                elif r > 3:
                    img[x][y] = np.array([1., 0, 0])


class Line:
    """Line has two ends (which are ether true end or bifurcation point
    and consists of points np.array([x, y])
    one pixel width
    bifurcation point on fingerprint splits line into three - not two"""

    def __init__(self):
        self._points = []
        self._mask = {}

    def add_point(self, p):
        self._points.append(p)
        self._mask[(p[0], p[1])] = True

    def ends(self):
        return self._points[0].copy(), self._points[-1].copy()

    def ends_directions(self, n=10):
        N = min(n, len(self._points))
        left = np.zeros([2])
        right = np.zeros([2])
        if N < 2:
            return None, None
        for i in range(1, N):
            left += self._points[i]
            right += self._points[-i - 1]
        dir = np.array([(left - (N - 1) * self._points[0]), (right - (N - 1) * self._points[-1])])
        dir /= np.linalg.norm(dir)
        return dir

    def points(self):
        return self._points.copy()

    def get_line_from_image(self, x, y, img, mask=None):
        """calculates line and returns mask and True on sucsess (actually previous mask with current line"""
        if mask is None:
            mask = np.zeros(img.shape)
        elif mask[x][y] or not img[x][y]:
            return mask, False

        wdt, htd = img.shape

        def get_neigh(x, y):
            ney = []
            cnt = 0
            for i in range(-1, 2):
                if 0 <= x + i < wdt:
                    for j in range(-1, 2):
                        if 0 <= y + j < htd and img[x + i][y + j] and \
                                (not self._mask.get((x + i, y + j))) and (i or j) :
                            cnt += 1
                            if not mask[x + i][y + j]:
                                ney.append(np.array([x + i, y + j]))

            return ney, cnt
        
        ney, cnt = get_neigh(x, y)
        if not (0 < cnt < 3 and  0 < len(ney) < 3) :
            #print(ney)
            return mask, False
        mask[x][y] = 1
        self.add_point(np.array([x, y]))
        r = ney[0]
        if len(ney) == 2:
            l = ney[1]
            while True:
                mask[l[0]][l[1]] = 1
                self.add_point(l)
                ney, cnt = get_neigh(l[0], l[1])
                if cnt != 1 or not ney:
                    #print(cnt, ney)
                    break
                l = ney[0]
            self._points.reverse()

        while True:
            mask[r[0]][r[1]] = 1
            self.add_point(r)
            ney, cnt = get_neigh(r[0], r[1])
            if cnt != 1 or not ney:
                #print(cnt, ney)
                break
            r = ney[0]
        return mask, True

    def draw(self, img, val = (255,)): #array of chennels)
       c = len(val)
       for x, y in self._points:
           for cc in range(c):
               img[x][y*c + cc] = val[cc]

    def draw_end(self, img, val = (255,)):
        c = len(val)
        ij = [(-2, 0), (-1, -1), (0, -2), (1, -1), (2, 0), (1, 1), (0, 2), (-1, 1)]
        for x, y in self.ends():
            for i, j in ij:
                if 0 <= x + i< img.shape[0] and 0 <= y + j < img.shape[1] / c:
                    for cc in range(c):
                        img[x + i][(y + j) * c + cc] = val[cc]

    def join(self, line, l1_end, l2_end): # 0 end means _points[0] other - points[-1]
        res = Line()
        l1 = self
        l2 = line
        pts1 = l1.points()
        pts2 = l2.points()
        if l1_end == 0:
            pts1.reverse()
        if l2_end != 0:
            pts2.reverse()

        x1, y1 = p1 = pts1[-1]
        x2, y2 = pts2[0]
        dx = x2 - x1
        dy = y2 - y1

        pts12 = []
        if dx or dy:
            ddx = ddy = 1

            n = max(dx, dy)

            if dx > dy:
                ddy = dy/dx
            else:
                ddx = dx/dy

            dd = np.array([ddx, ddy])
            t = p1 + dd
            for i in range(n):
                pts12.append(t.astype(int))
                t += dd
        pts1.extend(pts12)
        pts1.extend(pts2)
        for p in pts1:
            res.add_point(p)
        return res

def get_lines(img):
    mask = None
    lines = []
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if (img[x][y] > 0):
                line = Line()
                mask, res = line.get_line_from_image(x, y, img, mask)
                if res:
                    lines.append(line)
    return lines, mask


def plot_lines(rng, img, lines, chanels = 3):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255),
              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128),
              (128, 255, 0), (128, 0, 255), (0, 128, 255), (255, 128, 0), (255, 0, 128), (0, 255, 128)] if chanels == 3 else \
        [(255,), (200,), (150,), (100,), (50,)]
    res = np.zeros((img.shape[0], img.shape[1] * chanels))
    w = png.Writer(img.shape[0], img.shape[1], greyscale=(chanels == 1))
    for i in rng:
        for l in lines:
            cl = random.choice(colors)
            l.draw(res, cl)
            l.draw_end(res, cl)

        with open("lines{}.png".format(i), 'wb') as f:
            w.write(f, res)  # .reshape(-1, 512 * 3))

# mark_points(img, 5)

# plt.subplot(1, 2, 1)
# plt.imshow(img0, 'gray')
# plt.title("img_tr")
#
# plt.subplot(1, 2, 2)


lines, mask = get_lines(img)

plot_lines(range(3), img, lines, 3)

for c in range(10):
    l10 = random.choice(lines)
    #lines.remove(l10)
    l20 = random.choice(lines)
    #lines.remove(l20)
    l30 = l10.join(l20, 0, 1)
plot_lines(range(3, 5), img, lines, 3)

points_xl = []
points_yl = []
points_xr = []
points_yr = []
for l in lines:
    e1, e2 = l.ends()
    points_xr.append(e1[0])
    points_yr.append(e1[1])
    points_xl.append(e2[0])
    points_yl.append(e2[1])



plt.imshow(mask, 'gray')
plt.title("img_points")
plt.plot(points_yr, points_xr, 'bx')
plt.plot(points_yl, points_xl, 'rx')
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
