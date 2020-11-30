import cv2
import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.001


def show_img(imgs):
    for i in range(len(imgs)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i], cmap='gray')
    plt.show()


def get_radius(vec):
    vec_f = []
    for i in range(vec.shape[0]):
        vec_f[i:i] = [np.median(vec[i:i + 5])]
    radius = np.argmin(vec_f) - np.argmax(vec_f)
    print(abs(radius))
    return abs(radius)


def Gaussian2D(shape, miu_1, miu_2, sigma_1, sigma_2):
    PSF = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            PSF[i, j] = np.exp(-0.5 * ((i - miu_1)**2 / sigma_1**2 +
                                       (j - miu_2)**2 / sigma_2**2))

    return PSF


if __name__ == '__main__':

    img = cv2.imread('./image.jpg')

    # parameter estimate
    b_img, g_img, r_img = cv2.split(img)
    b_img_s = np.fft.fftshift(np.fft.fft2(b_img))

    b_lap = cv2.Laplacian(b_img, -1, ksize=1)
    b_lap_s = np.fft.fftshift(np.fft.fft2(b_lap))
    b_s_corr = np.abs(b_lap_s)**2

    vec_horizontal = b_s_corr[b_img.shape[0] // 2]
    r_horizontal = get_radius(vec_horizontal)

    vec_vertical = b_s_corr[:, b_img.shape[1] // 2]
    r_vertical = get_radius(vec_vertical)
    #plt.plot(np.log(vec_vertical))
    #plt.show()

    # get PSF

    print(b_img.shape)
    miu_1 = b_img.shape[0] // 2
    miu_2 = b_img.shape[1] // 2
    sigma_1 = r_horizontal / 10
    sigma_2 = r_vertical / 5
    print(miu_1, miu_2, sigma_1, sigma_2)
    PSF = Gaussian2D(b_img.shape, miu_1, miu_2, sigma_1, sigma_2)
    """
    PSF = np.zeros(tuple(list(b_img.shape) + [3]))
    center_point = (PSF.shape[1] // 2, PSF.shape[0] // 2)
    #cv2.circle(PSF, center_point, radius, (0, 0, 1), -1)
    axis = (r_horizontal, r_vertical)  #(8, 6)
    cv2.ellipse(PSF, center_point, axis, 0, 0, 360, (0, 0, 10), -1)
    PSF = cv2.split(PSF)[2]
    """

    PSF_s = np.fft.fftshift(np.fft.fft2(PSF))

    # deconvolution

    b_recover_s = np.conj(PSF_s) * b_img_s
    b_recover_s = b_recover_s / (b_img_s * b_img_s + epsilon)
    """
    b_recover_s = b_img_s / PSF_s
    """
    b_recover = np.abs(np.fft.ifft2(np.fft.fftshift(b_recover_s)))
    plt.imshow(b_recover, cmap='gray')
    plt.show()

    imgs = [
        b_img,  #1
        np.log(np.abs(b_img_s)),  #2
        np.log(np.abs(b_lap_s)),  #3
        PSF,  #4
        np.log(np.abs(PSF_s)),  #5
        np.log(np.abs(b_recover_s)),  #6
        b_recover  #7
    ]

    show_img(imgs)

    pass