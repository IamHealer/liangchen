import numpy as np
def rotateFlip(image,option):
    h, w, _ = image.shape
    if option in ['LEFT','RIGHT']:
        temp = np.zeros((w, h, 3), np.uint8)  # null image with inverted dimensions
        if option == 'LEFT':
            for i in range(0, w):
                temp[w - i - 1, :, :] = image[:, i, :]
        elif option == 'RIGHT':
            for j in range(0, h):
                temp[:, h - j - 1, :] = image[j, :, :]
    elif option in ['HORIZONTAL','VERTICAL']:
        temp = np.zeros((h, w, 3), np.uint8)
        if option == 'HORIZONTAL':
            for i in range(0, w):
                temp[:, i, :] = image[:, w - i - 1, :]
        elif option == 'VERTICAL':
            for j in range(0, h):
                temp[j, :, :] = image[h - j - 1, :, :]

    return temp


# def flip(image, option):
#     h, w, _ = image.shape
#     temp = np.zeros((h, w, 3), np.uint8)
#     if option == 'HORIZONTAL':
#         for i in range(0, w):
#             temp[:, i, :] = image[:, w - i - 1, :]
#     elif option == 'VERTICAL':
#         for j in range(0, h):
#             temp[j, :, :] = self.currentImage.image[h - j - 1, :, :]
