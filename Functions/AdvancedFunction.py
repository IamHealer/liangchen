import cv2
import skimage
from sklearn.cluster import KMeans

from numpy import linalg as LA
import numpy as np


def pixelate(img, w, h):

    height, width = img.shape[:2]

    # Resize input to "pixelated" size
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Initialize output image
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def colorClustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]

    return imgC


def segmentImgClrRGB(img, k):

    imgC = np.copy(img)

    h = img.shape[0]
    w = img.shape[1]
    print(img.shape)
    imgC.shape = (img.shape[0] * img.shape[1], 3)

    # 5. Run k-means on the vectorized responses X to get a vector of labels (the clusters);
    #
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_

    # 6. Reshape the label results of k-means so that it has the same size as the input image
    #   Return the label image which we call idx
    kmeans.shape = (h, w)

    return kmeans

def kMeansImage(image, k):
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)


import numpy as np
import cv2


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""


    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def augContrast(src):
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
        return src
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。

    else:
        max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

        # 去掉分位值区间之外的值
        src[src >= max_percentile_pixel] = max_percentile_pixel
        src[src <= min_percentile_pixel] = min_percentile_pixel

        # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
        out = np.zeros(src.shape, src.dtype)
        cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

        return out


def get_lightness(src):
    # 计算亮度
    if len(src.shape)==3:
        hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    else:
        hsv_image = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV)

    lightness = hsv_image[:, :, 2].mean()

    return lightness

