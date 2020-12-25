from typing import List, Any

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
DEBUG = False


class CoinDetector_tm():
    ''' coin detector with template matching '''

    def __init__(self, coinTemplates):
        self.coinTemplateFiles = coinTemplates.copy()
        self.img_templates = []
        self.scale = np.arange(1, 0.0, -0.1)
        number_of_processes = 4
        self.pool = multiprocessing.Pool(number_of_processes)


        self.im_t_resized = {}
        for t in self.coinTemplateFiles:
            img1 = cv2.imread(t)
            img1 = preprocessImage(img1, resize = False)
            self.img_templates.append(img1)
            self.im_t_resized[t] = dict()
            h_t, w_t = img1.shape[0:2]
            for s in self.scale:
                im_t_resized = cv2.resize(img1, (int(w_t * s), int(h_t * s)))
                self.im_t_resized[t][s] = im_t_resized


    def findCoins(self, target, threshold=0.7, plot=False):
        ''' Find coins in target image
            target: target image
        '''
        h,w = target.shape[0:2]
        target = preprocessImage(target, applyBlock=True)

        xc = []
        yc = []
        input = []
        for t in self.coinTemplateFiles:
            input.append((t, target, threshold, self.scale, self.im_t_resized))
        results = []
        for i in input:
            r = _findCoinOneTemplate(i)
            results.append(r)

        #results = self.pool.map(_findCoinOneTemplate, input)
        for x,y in results:
            xc += x
            yc += y
        xc = np.array(xc)*w
        yc = np.array(yc)*h
        return xc,yc

def _findCoinOneTemplate(input):
    t, target, threshold, scale, im_t_resized = input
    plot = False
    xc = []
    yc = []
    boxes_all = []
    value_all = []
    for s in scale:
        res = cv2.matchTemplate(target, im_t_resized[t][s], cv2.TM_CCOEFF_NORMED)
        w, h = im_t_resized[t][s].shape[0:2]
        if w < 10 or h < 10:
            continue
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # top_left = max_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        # cv2.rectangle(target, top_left, bottom_right, 255, 2)
        loc = np.where(res >= threshold)
        value_cand = res[loc]
        boxes_cand = np.array([[pt[0], pt[1], pt[0] + w, pt[1] + h] for pt in zip(*loc[::-1])])
        boxes, value = non_max_suppression_fast(boxes_cand, value_cand, 0.8)
        target_plot = target.copy()
        for box, v in zip(boxes, value):
            boxes_all.append(box)
            value_all.append(v)
            cv2.rectangle(target_plot, tuple(box[0:2]), tuple(box[2:4]), 255, 2)
        if DEBUG:
            plt.clf()
            plt.subplot(221)
            plt.imshow(target_plot)
            plt.subplot(222)
            plt.imshow(im_t_resized[t][s])
            plt.title('scale %1.2f' % s)
            plt.subplot(223)
            plt.imshow(res, cmap='gray')
            pass


    boxes, value = non_max_suppression_fast(np.array(boxes_all),
                                            np.array(value_all),
                                            0.8)
    for box in boxes:
        xc.append(int((box[0] + box[2]) / 2)/target.shape[1])
        yc.append(int((box[1] + box[3]) / 2)/target.shape[0])
    return xc, yc


class CoinDetector():
    ''' wrapper class for coin detection '''

    def __init__(self, coinTemplates):
        '''
        :param coinTemplates:  list of coin template file names for matching
        '''
        self.coinTemplateFiles = coinTemplates.copy()
        self.coinTemplates = []
        self.img_templates = []
        self.sift = cv2.SIFT_create(nfeatures=0,
                                    nOctaveLayers=3,
                                    contrastThreshold=0.01,
                                    edgeThreshold=10)
        self.distanceRatio = 0.7

        # detect keypoints for coin templates
        self.kp_template = []
        self.des_template = []
        for t in self.coinTemplateFiles:
            img1 = cv2.imread(t)
            gray1 = preprocessImage(img1)
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            self.kp_template.append(kp1)
            self.des_template.append(des1)
            self.img_templates.append(img1)

    def findCoins(self, target, maxMatches=10, plot=False):
        ''' Find coins in target image
            target: target image
            maxMatches: max match per template
            plot: plot debug image if True
        '''

        x_all = []
        y_all = []
        # find keypoints of target image
        gray_target = preprocessImage(target, applyBlock=True)
        kp_target, des_target = self.sift.detectAndCompute(gray_target, None)
        self.kp_target = kp_target
        self.des_target = des_target
        self.matches = []
        for kp, des in zip(self.kp_template, self.des_template):
            # feature matching
            matches = findMatches(des, des_target, maxMatches, mode='knn', distanceRatio=self.distanceRatio)
            x, y = coinLocation(kp_target, matches)
            x_all += x
            y_all += y
            self.matches.append(matches)

        unique_xy = []
        for x, y in zip(x_all, y_all):
            if (x, y) not in unique_xy:
                unique_xy.append((x, y))
        x_all = [x for x, y in unique_xy]
        y_all = [y for x, y in unique_xy]
        return x_all, y_all


def non_max_suppression_fast(boxes, value, overlapThresh):
    '''

    :param boxes: numpy array of [x1,y1,x2,y2]. (x1,y1) is top left corner. (x2, y2) is bottom right corner.
    :param value: confidence value associated with the box. Highest value pick first
    :param overlapThresh:
    :return:
    '''
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(value)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), value[pick]


def blockImageArea(gray, x, y, value=0.):
    ''' Replace a patch of area with value '''
    x = [int(gray.shape[1] * x[0]), int(gray.shape[1] * x[1])]
    y = [int(gray.shape[0] * y[0]), int(gray.shape[0] * y[1])]
    gray[y[0]:y[1], x[0]:x[1]] = value
    return gray


def preprocessImage(img, applyBlock=False, useGray=True, resize = True):
    ## convert to hsv
    if useGray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask of yellow (15,0,0) ~ (36, 255, 255)
        mask = cv2.inRange(img, (15, 120, 120), (36, 255, 255))
        img = cv2.bitwise_and(img, img, mask=mask)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.Canny(img, 10, 1000)

    if applyBlock:
        # remove coin icon
        x = [0.75, 0.81]
        y = [0.8, 0.95]
        img = blockImageArea(img, x, y, 0)
        # remove inventory box
        x = [0., 0.15]
        y = [0.3, 0.5]
        img = blockImageArea(img, x, y, 0)
        # remove pumpkin
        x = [0.85, 1]
        y = [0.65, 0.8]
        img = blockImageArea(img, x, y, 0)

    if resize:
        img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)) )

    return img


def findMatches(des1, des2, maxMatches, mode='simple', distanceRatio=0.75):
    ''' Find matches '''
    if mode == 'simple':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[0:maxMatches]
        return matches
    elif mode == 'knn':
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < distanceRatio * n.distance:
                good.append(m)
        good = sorted(good, key=lambda x: x.distance)
        good = good[0:maxMatches]

        return good
    elif mode == 'flann':
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        # ratio test as per Lowe's paper
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        good = sorted(good, key=lambda x: x.distance)
        good = good[0:maxMatches]
        return good


def coinLocation(kp2, matches):
    ''' return x,y locations of keypoints '''
    # query is 1, train is 2
    loc = [kp2[m.trainIdx].pt for m in matches]
    x = [l[0] for l in loc]
    y = [l[1] for l in loc]
    return x, y


def findCoins(templates, target, maxMatches=10, plot=False):
    '''
        template: coin templates (list of file names or
        target: target image
        maxMatches: max match per template
    '''
    img2 = cv2.imread(target)
    fignum = 0
    x_all = []
    y_all = []
    for t in templates:
        img1 = cv2.imread(t)
        sift = cv2.SIFT_create(nfeatures=0,
                               nOctaveLayers=5,
                               contrastThreshold=0.05)
        gray1 = preprocessImage(img1)
        gray2 = preprocessImage(img2, applyBlock=True)
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # feature matching
        matches = findMatches(des1, des2, maxMatches, mode='knn')
        x, y = coinLocation(kp2, matches)
        x_all += x
        y_all += y
        if plot:
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
            # for m in matches:
            #     print(m.distance)

            # plt.figure(fignum)
            # fignum += 1

            plt.figure(fignum)
            fignum += 1
            plt.imshow(img3)

            # img1 = cv2.drawKeypoints(gray1, kp1, img1)
            # plt.imshow('image', img1)
    if plot:
        plt.figure(fignum)
        fignum += 1
        plt.imshow(gray2)
        xc, yc = [0.5, 0.75]
        x = np.mean(x_all)
        y = np.mean(y_all)
        print(x, y)
        plt.plot(x, y, marker='x', color='white')
        plt.plot(xc * gray2.shape[1], yc * gray2.shape[0], marker='x', color='green')


if __name__ == "__main__":
    tt = r'C:\Users\ctawo\PycharmProjects\autoclicker\mm2\coin%d.png'
    templates = [tt % i for i in range(1, 6)]
    filename2 = r'C:\Users\ctawo\PycharmProjects\autoclicker\mm2\img_040.png'

    findCoins(templates, filename2, plot=True)
    plt.show()
