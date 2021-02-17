import cv2
import numpy as np
import matplotlib.pyplot as plt

# Background: Sky and stars.
# Foreground: Buildings, trees, etc.

PIC_SHAPE = (3376, 6000, 3)
GROUP_SIZE = 16
START_INDEX = 106
END_INDEX = 325


def cv_close(img, kernel):
    tmp = cv2.dilate(img, kernel)
    return cv2.erode(tmp, kernel)


def cv_open(img, kernel):
    tmp = cv2.erode(img, kernel)
    return cv2.dilate(tmp, kernel)


def read_images(start_index: int, end_index: int, group_index: int):
    num_pics = end_index - start_index
    imgs = np.ndarray([num_pics, *PIC_SHAPE], dtype=np.uint8)
    imgs_gray = np.ndarray([num_pics, *PIC_SHAPE[:2]], dtype=np.uint8)

    def file_name(index: int):
        if (group_index >= 0):
            i = index
            if i + 1 < 10:
                str_i = '00' + str(i + 1)
            elif 10 <= i + 1 and i + 1 < 100:
                str_i = '0' + str(i + 1)
            else:
                str_i = str(i + 1)
            return './img/DSC08' + str_i + '.JPG'
        else:
            return './tmp/tmp_' + str(index) + '.jpg'

    for i in range(start_index, end_index):
        tmp = cv2.imread(file_name(i))
        if tmp is not None:
            imgs[i - start_index] = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            imgs_gray[i - start_index] = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    return imgs, imgs_gray


def generate_background_mask(image_gray, image):

    R_INDEX_THRESHOLD = 4
    GRAY_LAPLACE_THRESHOLD = 5
    KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    KERNEL_21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    KERNEL_41 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    fore_mask_0 = np.zeros_like(image_gray)
    fore_mask_1 = np.zeros_like(image_gray)
    bg_mask = np.zeros_like(image_gray)

    # red map
    img_R = image[:, :, 0].astype(np.float)
    img_G = image[:, :, 1].astype(np.float)
    epsilon = 1
    img_R_map = (img_R) / (img_G + epsilon)
    max_value = img_R_map.max()
    img_R_map = img_R_map / max_value * 255
    img_R_map = img_R_map.astype(np.uint8)
    fore_mask_0[img_R_map > R_INDEX_THRESHOLD] = 1
    fore_mask_0 = cv_close(fore_mask_0, KERNEL_21)

    # edge detect
    img_med = cv2.medianBlur(image_gray, 9)
    img_med = cv2.medianBlur(img_med, 9)
    gray_lap = cv2.Laplacian(img_med, cv2.CV_16S, ksize=3)
    gray_lap = cv2.convertScaleAbs(gray_lap)
    gray_lap[gray_lap < GRAY_LAPLACE_THRESHOLD] = 0
    gray_lap[gray_lap > GRAY_LAPLACE_THRESHOLD] = 1
    gray_lap_open = cv_open(gray_lap, KERNEL_3)
    fore_mask_1 = cv_close(gray_lap_open, KERNEL_41)

    # get background mask
    fore_mask = fore_mask_0 + fore_mask_1
    fore_mask[fore_mask > 1] = 1
    bg_mask_ini = 1 - fore_mask
    mask_uint8 = bg_mask_ini.astype(np.uint8) * 255  # chenge dtype to uint8
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)  # find connected areas
    areas = [cv2.contourArea(cnt) for cnt in contours]  # calc squares
    tar_index = np.argsort(areas)[-1]  # select the biggest one
    cv2.drawContours(bg_mask, contours, tar_index, 1, cv2.FILLED)

    bg_c3_mask = cv2.merge([bg_mask.copy(), bg_mask.copy(), bg_mask.copy()])
    return bg_mask, bg_c3_mask


def gen_cache_trail(start_index: int, end_index: int, group_index: int):

    # get images
    imgs, imgs_gray = read_images(start_index, end_index, group_index)

    # generate background mask for all channels
    bg_mask, bg_c3_mask = generate_background_mask(imgs_gray[0], imgs[0])

    # get masked images and gray images
    imgs_bg = imgs * bg_c3_mask
    bgs_gray = imgs_gray * bg_mask

    # pick the brightest pix among all pics for each pixel.
    res_bg = np.max(imgs_bg, axis=0)

    # compose foreground and background
    fore_c3_mask = 1 - bg_c3_mask
    res_fore = imgs[0] * fore_c3_mask
    res = res_fore + res_bg
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    # save as file
    if group_index >= 0:
        cv2.imwrite('./tmp/tmp_' + str(group_index) + '.jpg', res)
    else:
        cv2.imwrite('./res.jpg', res)
    pass


def gen_trail(start_index: int, end_index: int):
    pic_num = end_index - start_index
    group_num = (pic_num // GROUP_SIZE) + 1

    for i in range(group_num):
        gen_cache_trail(
            start_index + i*GROUP_SIZE,
            start_index + min(pic_num, (i + 1) * GROUP_SIZE), i)

    gen_cache_trail(0, group_num, -1)
    pass


if __name__ == "__main__":
    gen_trail(START_INDEX, END_INDEX)
    pass
