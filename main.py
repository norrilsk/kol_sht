import copy
import random
import time

import cv2 as cv
import numpy as np
import png
from skimage.morphology import skeletonize, erosion


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


class ROI:
    def __init__(self, img, size = 32):
        self.size = size
        area0 = np.zeros(((img.shape[0] + size - 1)//size + 2, (img.shape[1] + size - 1)//size + 2))
        for i in range(1, area0.shape[0] - 2): # NIST only
            x = (i - 1)*size
            xdx = min(x + size, img.shape[0])
            for j in range(1, area0.shape[1] - 1):
                y = (j - 1)*size
                ydy = min(y + size, img.shape[1])
                val = np.mean(img[x:xdx, y:ydy])
                if val > 0.3:
                    area0[i][j] = 1
        self.area = erosion(area0)[1:-1, 1:-1]

    def ok(self, x, y = None):
        x, y = x if y is None else x, y
        u = min(self.area.shape[0], max(0, int(x)//self.size))
        v = min(self.area.shape[1], max(0, int(y)//self.size))
        return bool(self.area[u][v])

    def get_key(self, a1, a3):
        res = [0]* ((self.area.shape[0] - 2) * (self.area.shape[1] - 2))
        points = [*a1, *a3]
        ll = self.area.shape[1] -1
        for x, y in points:
            if self.ok(x, y):
                u = min(self.area.shape[0], max(0, int(x) // self.size))
                v = min(self.area.shape[1], max(0, int(y) // self.size))
                res[(u - 1)*ll + (v - 1)] += 1
        key = 0
        for i, r in enumerate(res):
            if self.ok((i // ll + 1.5)*self.size, (i % ll + 1.5)*self.size):
                key *= 2
                key += int(r > 0)
        return key


class Line:
    """Line has two ends (which are ether true end or bifurcation point
    and consists of points np.array([x, y])
    one pixel width
    bifurcation point on fingerprint splits line into three - not two"""

    def __init__(self):
        self._points = []
        self._mask = {}

    def size(self):
        return len(self._points)

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
        dir = -np.array([(left - (N - 1) * self._points[0]), (right - (N - 1) * self._points[-1])])
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


    def extend_to(self, end, x2, y2, shape=None):
        if end == 0:
            self._points.reverse()
        if shape:
            x2 = min(max(0, x2), shape[0] - 1)
            y2 = min(max(0, y2), shape[0] - 1)
        x1, y1 = p1 = self._points[-1]
        dx = x2 - x1
        dy = y2 - y1
        pts12 = []
        if dx or dy:
            ddx = 1 if dx > 0 else -1
            ddy = 1 if dy > 0 else -1

            n = max(abs(dx), abs(dy))

            if abs(dx) > abs(dy):
                ddy = dy / abs(dx)
            else:
                ddx = dx / abs(dy)

            dd = np.array([ddx, ddy])
            t = p1 + dd
            for i in range(n):
                pts12.append(t.astype(int))
                t += dd
        self._points.extend(pts12)

    def join(self, line, l1_end, l2_end): # 0 end means _points[0] other - points[-1]
        res = Line()
        l1 = copy.deepcopy(self)
        l2 = line

        pts2 = l2.points()

        if l2_end != 0:
            pts2.reverse()

        l1.extend_to(l1_end, pts2[0][0], pts2[0][1])
        pts1 = l1.points()
        pts1.extend(pts2)

        for p in pts1:
            res.add_point(p)
        return res

    def split_by(self, ind):
        line1 = Line()
        line2 = Line()
        for p in self._points[:ind]:
            line1.add_point(p)
        for p in self._points[ind:]:
            line2.add_point(p)
        return line1, line2

    def get_nearest(self, point):
        rpt = None
        rind = None
        d = 1e18
        for ind, pt in enumerate(self._points):
            dd = pt - point
            l = dd.dot(dd)
            if l < d:
                d = l
                rpt = pt
                rind = ind
        return rpt, rind

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


def plot_lines(rng, img, lines, chanels=3, additional=None):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255),
              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128),
              (128, 255, 0), (128, 0, 255), (0, 128, 255), (255, 128, 0), (255, 0, 128), (0, 255, 128)] if chanels == 3 else \
        [(255,), (200,), (150,), (100,), (50,)]
    res = np.zeros((img.shape[0], img.shape[1] * chanels))
    w = png.Writer(img.shape[0], img.shape[1], greyscale=(chanels == 1))
    for i in rng:
        for j, l in enumerate(lines):
            cl = random.choice(colors)
            l.draw(res, cl)
            l.draw_end(res, cl)
            if additional:
                additional[2*j].draw(res, cl)
                additional[2*j + 1].draw(res, cl)
        with open("lines{}.png".format(i), 'wb') as f:
            w.write(f, res)  # .reshape(-1, 512 * 3))

class End:
    def __init__(self, _point=None, _direction=None, _line=None, _endi=None):
        self.point = _point
        self.direction = _direction
        self.line = _line
        self.endi = _endi

    def extend(self, shape,  k = 15):
        tl = Line()
        tl.add_point(self.point)
        x = (self.point + self.direction * k).astype(int)
        x[0] = min(shape[0] - 1, max(0, x[0]))
        x[1] = min(shape[1] - 1, max(0, x[1]))
        tl.extend_to(1, x[0], x[1])
        return tl


def draw_ptp_line(img, p1, p2, val = (1,)):
    w = img.shape[0]
    h = img.shape[1]
    dx, dy = p2 - p1
    if dx or dy:
        ddx = 1 if dx > 0 else -1
        ddy = 1 if dy > 0 else -1

        n = max(abs(dx), abs(dy))

        if abs(dx) > abs(dy):
            ddy = dy / abs(dx)
        else:
            ddx = dx / abs(dy)

        dd = np.array([ddx, ddy])
        t = p1 + dd
        for i in range(n):
            ti = t.astype(int)
            if 0 <= ti[0] < w and 0 <= ti[1] < h:
                cn = len(val)
                if cn ==1:
                    img[ti[0]][ti[1]] = val[0]
                for cc in range(cn):
                    img[ti[0]][ti[1]*cn + cc] = val[cc]
            t += dd



def join_interesting_ends(lines, roi, mask, TRESHHOLD = 17):
    lines0 = copy.deepcopy(lines)
    lines = []
    for l in lines0:
        #l.erode_ends(2)
        if(l.size() > 3):
           # l.dilate_ends(3)
            lines.append(l)
            l.draw(mask, (1,))

    ans = [] # (np(x, y), np(dx, dy), line, ei1)
    for l1i, l1 in enumerate(lines):
        if not l1.size():
            continue
        for ei1 in range(2):
            p1 = l1.ends()[ei1]
            cnt = 0
            for l2i, l2 in enumerate(lines):
                if not l2.size():
                    continue
                for ei2 in range(2):
                    p2 = l2.ends()[ei2]
                    dd = (p1 - p2)
                    if dd.dot(dd) <= 4:
                        cnt += 1

            if cnt < 3:
                ans.append(End(p1, l1.ends_directions(30)[ei1], l1, ei1))

    marked = {}
    for i, e1 in enumerate(ans):
        if not roi.ok(e1.point[0], e1.point[1]):
            continue
        ends = []

        for j, e2 in enumerate(ans):
            if j == i:
                continue
            dd = e2.point - e1.point
            ddn = dd / np.linalg.norm(dd)
            ddl =dd.dot(dd)
            if ddl < TRESHHOLD* TRESHHOLD and e1.direction.dot(e2.direction) < 0:
                val = ddn.dot(e1.direction) - ddn.dot(e2.direction)
                if val > 0.7:
                    ends.append((val + 1./ddl, -ddl, j))

        if ends:
            tos = sorted(ends)[:3]
            for to in tos:
                j = to[2]
                en = ans[j]
                l = Line()
                l.add_point(e1.point)
                l.extend_to(1, en.point[0], en.point[1], mask.shape)
                con = False
                for p in l.points()[1:-1]:
                    if mask[p[0]][p[1]]:
                        con = True
                        break
                if con or marked.get((i, j)):
                    continue
                #print(i, j, to[0], to[1])
                marked[(i, j)] = True
                marked[(j, i)] = True
                l.draw(mask, (1,))

        for li in lines:
            p, ind = li.get_nearest(e1.point)
            dd = p - e1.point

            ddl = dd.dot(dd)
            if 2 > ddl or ddl > TRESHHOLD * TRESHHOLD or ind < 7 or li.size() - ind < 7:
                continue
            ddn = dd / np.linalg.norm(dd)
            if ddn.dot(e1.direction) > 0.6:
                l = Line()
                l.add_point(e1.point)
                l.extend_to(1, p[0], p[1], mask.shape)
                if l.size() > li.size()/2 or l.size() > e1.line.size()/2:
                    continue
                con = False
                for p in l.points()[1:-1]:
                    if mask[p[0]][p[1]]:
                        con = True
                        break
                if con:
                    continue
                l.draw(mask, (1,))


    #
    # for i, e1 in enumerate(ans):
    #     line1 = e1.extend(mask.shape, TRESHLEN)
    #     pts = line1.points()
    #     for j, p in enumerate(pts[1:]):
    #         if mask[p[0]][p[1]]:
    #             for p0 in pts[1:j + 1]:
    #                 mask[p0[0]][p0[1]] = 1
    #                 break

    return mask


def lines_join_step(mask, roi, len_threshold, del_threshold=0, img_lines_range=None, closed_png_name=None):
    lines, mask= get_lines(mask)
    if img_lines_range:
        plot_lines(img_lines_range, mask, lines)
    mask1 = join_interesting_ends(lines, roi, np.zeros(mask.shape), len_threshold)
    mask1 = skeletonize(mask1)
    if closed_png_name:
        w = png.Writer(mask1.shape[0], mask1.shape[1], greyscale=True)
        with open(closed_png_name +".png", 'wb') as f:
            w.write(f, mask1 * 255)  # .reshape(-1, 512 * 3))
    if del_threshold:
        lines02, _ = get_lines(mask1)
        lines[:] = []
        for l in lines02:
            if l.size() >= del_threshold:
                lines.append(l)
        mask1.fill(0)
        for l in lines:
            l.draw(mask1)
    return mask1

def detect_points(mask):
    lines0, mask = get_lines(mask)
    lines = []
    for l in lines0:
        # l.erode_ends(2)
        if (l.size() > 3):
            # l.dilate_ends(3)
            lines.append(l)

    ans1 = []  # (np(x, y))
    ans3 = []  # (np(x, y))
    seen = {}
    for l1i, l1 in enumerate(lines):
        if not l1.size():
            continue
        for ei1 in range(2):
            p1 = l1.ends()[ei1]
            if seen.get((p1[0], p1[1])):
                continue
            cnt = 0
            for l2i, l2 in enumerate(lines):
                if not l2.size():
                    continue
                for ei2 in range(2):
                    p2 = l2.ends()[ei2]
                    dd = (p1 - p2)
                    if dd.dot(dd) <= 4:
                        cnt += 1
                        seen[(p2[0], p2[1])] = True

            if cnt == 1:
                ans1.append(p1)
            elif cnt == 3:
                ans3.append(p1)
    return ans1, ans3


def print_result(gray_img, pts1, pts2, roi: ROI = None, filename ='final.png'):
    res = np.zeros((gray_img.shape[0], gray_img.shape[1] * 3))
    pt1 = [tuple(x) for x in pts1]
    pt2 = [tuple(x) for x in pts2]
    for x in range(gray_img.shape[0]):
        for y in range(gray_img.shape[1]):
            col = None
            if roi.ok(x, y):
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if roi.ok(x + i, y + j):
                            if (x + i, y + j) in pt1:
                                col = (0, 0, 255)
                            elif (x + i, y + j) in pt2:
                                col = (0, 255, 0)
                col = col or (gray_img[x][y], gray_img[x][y], gray_img[x][y])
                if x % roi.size == 0 or y % roi.size == 0:
                    col = (col[0]//3, col[1]//3, col[2]//3)
            else:
                co = gray_img[x][y]//3
                col = (co, co, co)
            for c in range(3):
                res[x][y*3 + c] = col[c]

    w = png.Writer(gray_img.shape[0], gray_img.shape[1])
    with open(filename, 'wb') as f:
        w.write(f, res)

def get_key_from_print(img00, file_name = None):
    sigma = 2  # experimental solution

    img = cv.GaussianBlur(img00, (25, 25), sigmaX=sigma, sigmaY=sigma)
    img0 = 1 - cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, 17, 2)

    img = skeletonize(img0).astype(int)
    roi = ROI(img0)

    mask = lines_join_step(img, roi, 7, del_threshold= 6)  # ,img_lines_range=(1, 2), closed_png_name="closed_1")
    mask = lines_join_step(mask, roi, 12, del_threshold=12)  # , closed_png_name="closed_2")
    mask = lines_join_step(mask, roi,  5, del_threshold=0)  # , closed_png_name="closed_3")
    mask = lines_join_step(mask, roi,  15, del_threshold=15)  # , closed_png_name="closed_4")
    mask = lines_join_step(mask, roi, 5, del_threshold=0)  # , closed_png_name="closed_5")
    mask = lines_join_step(mask, roi, 10, del_threshold=10)  # , closed_png_name="closed_6")
    w = png.Writer(mask.shape[0], mask.shape[1], greyscale=True)
    a1, a3 = detect_points(mask)
    if file_name:
        print_result(img00, a1, a3, roi, file_name)

    return roi.get_key(a1, a3)


if __name__ == '__main__':
    start = time.time()
    PATH_TO_DATA = "./data/NIST/"
    PATH_TO_IMAGE = PATH_TO_DATA + "figs_0/f0017_06.png"
    # PATH_TO_IMAGE = PATH_TO_DATA + "figs_0/f0145_10.png"
    img00 = cv.imread(PATH_TO_IMAGE, cv.IMREAD_GRAYSCALE)
    key = get_key_from_print(img00)  # , 'final.png')
    print(time.time() - start, 's')
    print(key)