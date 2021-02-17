import cv2
import numpy as np
import matplotlib.pyplot as plt


def cv_close(img, kernel):
    tmp = cv2.dilate(img, kernel)
    return cv2.erode(tmp, kernel)


def cv_open(img, kernel):
    tmp = cv2.erode(img, kernel)
    return cv2.dilate(tmp, kernel)


PIC_SHAPE = (3376, 6000, 3)
GROUP_SIZE = 16
RED_FORE_THRESHOLD = 40


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


def gen_cache_trail(start_index: int, end_index: int, group_index: int):

    imgs, imgs_gray = read_images(start_index, end_index, group_index)

    bg_mask = np.zeros_like(imgs_gray[0])
    bg_mask[imgs[0, :, :, 0] < RED_FORE_THRESHOLD] = 1
    filter_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    bg_mask = cv_close(bg_mask, filter_kernel)
    bg_c3_mask = cv2.merge([bg_mask.copy(), bg_mask.copy(), bg_mask.copy()])

    imgs_bg = imgs * bg_c3_mask
    bgs_gray = imgs_gray * bg_mask
    brightest_index = np.argmax(bgs_gray, axis=0)
    res_bg = np.ndarray(list(PIC_SHAPE), dtype=np.uint8)
    for i in range(brightest_index.shape[0]):
        for j in range(brightest_index.shape[1]):
            res_bg[i, j] = imgs_bg[brightest_index[i, j], i, j]

    fore_c3_mask = 1 - bg_c3_mask
    res_fore = imgs[0] * fore_c3_mask
    res = res_fore + res_bg
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

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
    gen_trail(106, 326)
    pass
