import cv2
import matplotlib.pyplot as plt
import numpy as np


def derection_filter(height: int, width: int):

    d_filter = np.zeros((height, width, 3), np.uint8)
    DIRECTION_COEF = 0.05
    RADIUS = 5

    # caution: x=width, y=height
    center_point = [width // 2, height // 2]
    r_1_point = [width, (int)(height / 2 * (1 - DIRECTION_COEF))]
    r_2_point = [width, (int)(height / 2 * (1 + DIRECTION_COEF))]
    l_1_point = [0, (int)(height / 2 * (1 - DIRECTION_COEF))]
    l_2_point = [0, (int)(height / 2 * (1 + DIRECTION_COEF))]
    u_1_point = [(int)(width / 2 * (1 - DIRECTION_COEF)), 0]
    u_2_point = [(int)(width / 2 * (1 + DIRECTION_COEF)), 0]
    d_1_point = [(int)(width / 2 * (1 - DIRECTION_COEF)), height]
    d_2_point = [(int)(width / 2 * (1 + DIRECTION_COEF)), height]

    right_triangle = np.array([center_point, r_1_point, r_2_point])
    left_triangle = np.array([center_point, l_1_point, l_2_point])
    up_triangle = np.array([center_point, u_1_point, u_2_point])
    down_triangle = np.array([center_point, d_1_point, d_2_point])

    cv2.fillConvexPoly(d_filter, right_triangle, (255, 255, 255))
    cv2.fillConvexPoly(d_filter, left_triangle, (255, 255, 255))
    cv2.fillConvexPoly(d_filter, up_triangle, (255, 255, 255))
    cv2.fillConvexPoly(d_filter, down_triangle, (255, 255, 255))
    cv2.circle(d_filter, tuple(center_point), RADIUS, (0, 0, 0), -1)

    d_filter = cv2.cvtColor(d_filter, cv2.COLOR_RGB2GRAY)
    d_filter = (255 - d_filter) // 255
    return d_filter


def task_1(img):
    """
    Task 1: remove grids on the photo.
    """
    MIDDLE_VALUE = 50
    MIDDLE_VALUE_2 = 120
    KERNAL_RADIUM = 8

    # convert to binary image
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = img_gray.copy()
    mask[img_gray > MIDDLE_VALUE] = 1
    mask[img_gray <= MIDDLE_VALUE] = 0

    # open operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (KERNAL_RADIUM, KERNAL_RADIUM))
    mask_er1 = cv2.erode(mask, kernel)
    mask_dl1 = cv2.dilate(mask_er1, kernel)  # 先腐蚀再膨胀，开运算

    # find the most connected domain
    mask_uint8 = mask_dl1.astype(np.uint8) * 255  # 转换为uint8
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)  # 寻找连通域
    areas = [cv2.contourArea(cnt) for cnt in contours]  # 计算面积
    tar_index = np.argsort(areas)[-1]  # 面积最大的连通区域就是目标工件区域
    mask_max_contour = np.zeros([mask.shape[0], mask.shape[1]])
    cv2.drawContours(mask_max_contour, contours, tar_index, 1, cv2.FILLED)
    mask_max_contour = mask_max_contour * mask_dl1  # 保留mask_dl1的孔洞部分
    mask_max_contour = np.array(mask_max_contour, dtype=int)

    # frequency domain transformation
    mask_spectrum = np.fft.fftshift(np.fft.fft2(mask_max_contour * 255))
    dir_filter = derection_filter(*img_gray.shape)
    mask_spectrum_filtered = mask_spectrum * dir_filter

    mask_filtered = np.fft.ifft2(np.fft.fftshift(mask_spectrum_filtered))
    mask_filtered = abs(mask_filtered)
    mask_filtered = mask_filtered / mask_filtered.max() * 255
    mask_filtered = np.array(mask_filtered, dtype=np.uint8)
    mask_filtered[mask_filtered > MIDDLE_VALUE_2] = 255
    mask_filtered[mask_filtered <= MIDDLE_VALUE_2] = 0
    mask_filtered = mask_filtered // 255

    KERNAL_RADIUM_2 = 6
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (KERNAL_RADIUM_2, KERNAL_RADIUM_2))
    mask_er2 = cv2.erode(mask_filtered, kernel_2)
    mask_dl2 = cv2.dilate(mask_er2, kernel_2)  # 先腐蚀再膨胀，开运算

    KERNAL_RADIUM_3 = 6
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (KERNAL_RADIUM_3, KERNAL_RADIUM_3))
    mask_dl3 = cv2.dilate(mask_dl2, kernel_3)
    mask_er3 = cv2.erode(mask_dl3, kernel_3)  # 先膨胀再腐蚀，闭运算

    # put the mask on the image
    img_no_grid = img_gray * mask_er3
    img_no_grid = cv2.cvtColor(img_no_grid, cv2.COLOR_GRAY2RGB)

    # show results
    titles = [
        'Source image', 'Binarization and open operation', 'Find max contour',
        'Spectrum', 'Frequency domain filtering', 'IFFT', "Open operation",
        'Close openation', "Result"
    ]
    images = [
        img, mask_dl1, mask_max_contour, 20 * np.log(np.abs(mask_spectrum)),
        20 * np.log(np.abs(mask_spectrum_filtered)), mask_filtered, mask_dl2,
        mask_er3, img_no_grid
    ]

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

    plt.show()
    cv2.imwrite('./remove_grid.jpg', img_no_grid)

    pass


def task_2(img):

    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edge_output = cv2.Canny(gray, 50, 150)

    plt.imshow(edge_output)
    plt.show()
    cv2.imwrite('./edge.jpg', edge_output)

    pass


if __name__ == '__main__':

    img = cv2.imread('./img.jpg')
    task_1(img)
    img_no_grid = cv2.imread('./remove_grid.jpg')
    task_2(img_no_grid)

    pass