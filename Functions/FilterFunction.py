import cv2
import numpy as np
# details filter
def changeDilate(img, value):
    kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
    img = cv2.dilate(img, kernel_size, iterations=1)
    return img


def changeErode(img, value):
    kernel_size = cv2.getStructuringElement(cv2.MORPH_RECT, (value + 1, value + 1))
    img = cv2.erode(img, kernel_size)
    return img


def changeContrast(img, value):
    if value >= 1 and value <= 10:
        h, w, ch = img.shape
        img2 = np.zeros([h, w, ch], img.dtype)
        img = cv2.addWeighted(img, value, img2, 1 - value, 0)  # addWeighted函数说明如下

    return img


def changeBlur( img, value):
    """ inputs: img image and blur values.
        perform blur operation using opencv function,
        return  img.
    """
    kernel_size = (value + 1, value + 1)  # +1 is to avoid 0
    img = cv2.blur(img, kernel_size)
    return img


def changeThreshold(img, thresh, maxval,threshType):
    ori_img, img = cv2.threshold(img, thresh, maxval,threshType)
    return img


def changeBrightness(img, value):
    """
        This function will take an image (img) and the brightness
        value. It will perform the brightness change using OpenCv
        and after split, will merge the img and return it.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.int16(v)
    v += value
    v = np.clip(v, 0, 255)
    v = np.uint8(v)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def changeHue( img, value):
    """
        This function will take an image (img) and the brightness
        value. It will perform the brightness change using OpenCv
        and after split, will merge the img and return it.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = np.int16(h)
    h += value
    h = np.clip(h, 0, 255)
    h = np.uint8(h)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def changeSaturation(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.int16(s)
    s += value
    s = np.clip(s, 0, 255)
    s = np.uint8(s)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
