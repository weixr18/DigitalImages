import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

EYE_SIZE = 100


def get_eye_filter():

    eye_filter = np.zeros([EYE_SIZE, EYE_SIZE, 3], dtype=np.uint8)
    center_point = (EYE_SIZE // 2, EYE_SIZE // 2)
    r1 = int(EYE_SIZE * 0.45)
    r2 = int(EYE_SIZE * 0.35)
    cv2.circle(eye_filter, center_point, r1, (255, 255, 255), 4)
    cv2.circle(eye_filter, center_point, r2, (255, 255, 255), 4)

    return eye_filter


def get_eye_region(img):

    print("Finding eye region...")
    eye_filter = get_eye_filter()[:, :, 0]
    eye_filter = eye_filter.astype(np.float)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_s = img_hsv[:, :, 1]
    img_v = img_hsv[:, :, 2]
    img_v_l = cv2.Laplacian(img_v, -1)
    img_v_l = img_v_l.astype(np.float)

    detect = signal.convolve2d(img_v_l,
                               eye_filter,
                               mode='same',
                               boundary='fill')
    detect = detect * (img_s > 127)
    detect = detect / detect.max() * 255
    detect = detect.astype(np.uint8)

    detect_index = np.argwhere(detect > 127)
    U, L = np.min(detect_index, axis=0)
    D, R = np.max(detect_index, axis=0)

    return U, D, L, R


def remove_red_eye(img):
    # input img: RGB
    # output img_res: RGB
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img_h = img_hsv[:, :, 0]
    img_s = img_hsv[:, :, 1]

    red_rigion_0 = (img_h < 22.5) * (img_h > 0)
    red_region_1 = ((180 - img_h) < 22.5) * ((180 - img_h) > 0)
    red_region = red_rigion_0 + red_region_1
    pure_region = (img_s > 0.3 * 255)
    red_eye = red_region * pure_region

    img_hsv[red_eye] = 0.01 * 255
    img_res = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_res


if __name__ == '__main__':

    img = cv2.imread('./1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    U, D, L, R = get_eye_region(img)
    eye_region = img[U:D, L:R]
    plt.imshow(eye_region)
    plt.show()
    removed = remove_red_eye(eye_region)

    img_res = img.copy()
    img_res[U:D, L:R] = removed
    plt.imshow(img_res)
    plt.show()
    cv2.imwrite('./res.jpg', img_res)

    pass