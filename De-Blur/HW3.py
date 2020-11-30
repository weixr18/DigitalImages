import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img(imgs):
    if (len(imgs) <= 1):
        plt.imshow(imgs[0], cmap='gray')
        plt.show()
        return

    shape = ()
    if (len(imgs) <= 2):
        shape = (1, 2)
    elif (len(imgs) <= 3):
        shape = (1, 3)
    elif (len(imgs) <= 4):
        shape = (2, 2)
    elif (len(imgs) <= 6):
        shape = (2, 3)
    elif (len(imgs) <= 9):
        shape = (3, 3)
    else:
        return

    for i in range(len(imgs)):
        plt.subplot(shape[0], shape[1], i + 1)
        plt.imshow(imgs[i], cmap='gray')
    plt.show()


def Gaussian2D(shape, sigma_1, sigma_2):
    PSF = np.ones(shape)
    miu_1 = shape[0] // 2
    miu_2 = shape[1] // 2

    for i in range(shape[0]):
        PSF[i, :] *= np.exp(-0.5 * (i - miu_1)**2 / sigma_1**2)

    for j in range(shape[1]):
        PSF[:, j] *= np.exp(-0.5 * (j - miu_2)**2 / sigma_2**2)

    PSF = PSF / (2 * np.pi * sigma_1 * sigma_2)
    return PSF


def no_use_recover(img):
    ETA = 1.5
    GAUSS_SHAPE = (10, 10)
    RADIUS = 10

    # spectrum
    img_s = np.fft.fftshift(np.fft.fft2(img))

    # filter
    gauss_filter = Gaussian2D(img.shape, GAUSS_SHAPE[0], GAUSS_SHAPE[1])
    gauss_s = np.fft.fftshift(np.fft.fft2(gauss_filter))

    circle_filter = np.zeros(tuple(list(img.shape) + [3]))
    center_point = (img.shape[1] // 2, img.shape[0] // 2)
    cv2.circle(circle_filter, center_point, RADIUS, (1, 0, 0), -1)
    circle_filter, _, __ = cv2.split(circle_filter)

    # high frequncy promote
    img_s_L = gauss_filter * img_s
    img_s_H = img_s - img_s_L
    """
    1
    """
    recov_s = img_s_L + ETA * img_s_H
    # ifft
    recov = np.abs(np.fft.ifft2(np.fft.fftshift(recov_s)))
    # dynamic range adjust
    dynam_range = recov.max() - recov.min()
    recov = recov / dynam_range * 255
    recov = recov.astype(int)
    """
    2
    """
    img_L = np.abs(np.fft.ifft2(np.fft.fftshift(img_s_L)))
    img_H = np.abs(np.fft.ifft2(np.fft.fftshift(img_s_H)))
    recov_2 = img_L + ETA * img_H
    dynam_range = recov_2.max() - recov_2.min()
    recov_2 = recov_2 / dynam_range * 255
    recov_2 = recov_2.astype(int)

    # show_img([img, recov, recov_2])

    return recov


def wiener_deconv(f, PSF, K=0.01):
    # 维纳滤波

    input_fft = np.fft.fftshift(np.fft.fft2(f))
    PSF_FFT = np.fft.fftshift(np.fft.fft2(PSF))

    PSF_fft_1 = np.conj(PSF_FFT) / (np.abs(PSF_FFT)**2 + K)
    PSF_fft_1_show = np.log(np.abs(PSF_fft_1))

    res_spectrum = input_fft * PSF_fft_1
    result = np.fft.ifftshift(np.fft.ifft2(res_spectrum))
    result = np.abs(result)

    return result  #np.abs(result)


R = 5


def recover(img):

    img_pad = flip_image(img)

    PSF = np.zeros(tuple(list(img_pad.shape) + [3]))
    center_x = img_pad.shape[1] // 2
    center_y = img_pad.shape[0] // 2
    cv2.rectangle(PSF, (center_x - R, center_y - R),
                  (center_x + R, center_y + R), (1, 0, 0), -1)

    PSF = PSF[:, :, 0]
    PSF /= np.sum(PSF)
    res_pad = wiener_deconv(img_pad, PSF, 0.01)

    res = cut_image(res_pad)
    dynam_range = res.max() - res.min()
    res = res / dynam_range * 255
    res = res.astype(int)
    return res


def flip_image(img):

    img_hor = cv2.flip(img, 1)
    img_ver = cv2.flip(img, 0)
    img_180 = cv2.flip(img_ver, 1)

    FLIP_A = int(0.1 * img.shape[0])
    new_shape = list(img.shape)
    new_shape[0] += 2 * FLIP_A
    new_shape[1] += 2 * FLIP_A
    img_pad = np.zeros(tuple(new_shape))

    # 0, 0
    img_pad[:FLIP_A, :FLIP_A] = img_180[img.shape[0] - FLIP_A:,
                                        img.shape[1] - FLIP_A]
    # 0, 1
    img_pad[:FLIP_A, FLIP_A:-FLIP_A] = img_ver[-FLIP_A:, :]
    # 0, 2
    img_pad[:FLIP_A, -FLIP_A:] = img_180[-FLIP_A:, :FLIP_A]
    # 1, 0
    img_pad[FLIP_A:-FLIP_A, :FLIP_A] = img_hor[:, -FLIP_A:]
    # 1, 1
    img_pad[FLIP_A:-FLIP_A, FLIP_A:-FLIP_A] = img
    # 1, 2
    img_pad[FLIP_A:-FLIP_A, -FLIP_A:] = img_hor[:, :FLIP_A]
    # 2, 0
    img_pad[-FLIP_A:, :FLIP_A] = img_180[:FLIP_A, -FLIP_A:]
    # 2, 1
    img_pad[-FLIP_A:, FLIP_A:-FLIP_A] = img_ver[:FLIP_A, :]
    # 2, 2
    img_pad[-FLIP_A:, -FLIP_A:] = img_180[:FLIP_A, :FLIP_A]

    # show_img([img, img_pad])
    return img_pad


def cut_image(img):
    FLIP_A = int(img.shape[0] / 12)
    return img[FLIP_A:-FLIP_A, FLIP_A:-FLIP_A]


if __name__ == '__main__':

    img = cv2.imread('./image.jpg')

    b_img, g_img, r_img = cv2.split(img)

    for i in range(1):
        b_recov = recover(b_img)
        g_recov = recover(g_img)
        r_recov = recover(r_img)

        img_recov = cv2.merge((r_recov, g_recov, b_recov))
        img = cv2.merge((r_img, g_img, b_img))

        #imgs = [b_img, b_recov, g_img, g_recov, r_img, r_recov, img_recov]
        imgs = [img, img_recov]
        show_img(imgs)

    pass
