from scipy.interpolate import UnivariateSpline
import cv2
import numpy as np
from PIL import Image
from PIL.ImageFilter import EMBOSS

def changeEffect(ori_img, effect='Original'):
    img = ori_img
    if effect == 'Original':
        img = ori_img
    elif effect == 'Greyscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif effect == 'Brightness':
        img = cv2.convertScaleAbs(img, beta=50)
    elif effect == 'Darker':
        img = cv2.convertScaleAbs(img, beta=-50)
    elif effect == 'Sharp':
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    elif effect == 'Sepia':
        img = np.array(img, dtype=np.float64)  # converting to float to prevent loss
        img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
                                            [0.349, 0.686, 0.168],
                                            [0.393, 0.769,
                                             0.189]]))  # multipying image with special sepia matrix
        img[np.where(img > 255)] = 255  # normalizing values greater than 255 to 255
        img = np.array(img, dtype=np.uint8)

    elif effect == 'Foil':
        img = np.array(Image.fromarray(img).filter(EMBOSS))
    elif effect == 'Median':
        img = cv2.medianBlur(img, 41)
    elif effect == 'Gaussian':
        img = cv2.GaussianBlur(img, (41, 41), 0)
    elif effect == 'PencilColor':
        sk_gray, sk_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.1)
        img = sk_color
    elif effect == 'HDR':
        img = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    elif effect == 'Invert':
        img = cv2.bitwise_not(img)
    elif effect == 'Summer':
        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        img = cv2.merge((blue_channel, green_channel, red_channel))
    elif effect == 'Winter':
        increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
        blue_channel, green_channel, red_channel = cv2.split(img)
        red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
        img = cv2.merge((blue_channel, green_channel, red_channel))
    print('filter ', img.shape)
    return img

def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))