import cv2
import numpy as np
import matplotlib.pyplot as plt

SIZE = 30


def cv_close(img, kernel):
    tmp = cv2.dilate(img, kernel)
    return cv2.erode(tmp, kernel)


def cv_open(img, kernel):
    tmp = cv2.erode(img, kernel)
    return cv2.dilate(tmp, kernel)


if __name__ == "__main__":

    img = cv2.imread('./img/1.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # erode small circles
    img[img > 0] = 1
    img = img.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SIZE, SIZE))
    big_circle = cv_close(img, kernel)

    # get small circle area
    small_circle = big_circle - img
    small_circle[small_circle != 1] = 0
    filter_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    small_circle = cv_open(small_circle, filter_kernel)

    # get circle edges
    small_circle = cv2.dilate(
        small_circle,
        filter_kernel)  # dilate so that line could detach from circle.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    small_circle_edge = small_circle - cv2.erode(small_circle, small_kernel)
    edge_points = np.array(np.where(small_circle > 0))  # [2, n]
    edge_points = edge_points[::-1, :].T  # [n, 2]
    edge_points = edge_points.reshape([1, -1, 2])  # [1, n, 2]

    # get convex hull
    hull_img = np.zeros_like(small_circle)
    hull = cv2.convexHull(edge_points)
    length = len(hull)
    for i in range(len(hull)):
        cv2.line(hull_img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]),
                 (1, 1, 1), 2)
    cv2.fillPoly(hull_img, hull, (1, 1, 1))
    hull_img = hull_img + img

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(small_circle)
    plt.subplot(1, 3, 3)
    plt.imshow(hull_img)
    plt.show()

    small_circle = small_circle * 255
    big_circle = big_circle * 255
    hull_img = hull_img * 127

    cv2.imwrite('./res/small_circle.png', small_circle)
    cv2.imwrite('./res/big_circle.png', big_circle)
    cv2.imwrite('./res/divide_line.png', hull_img)
    pass
