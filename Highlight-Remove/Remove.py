import cv2
import numpy as np
import matplotlib.pyplot as plt

T1 = 230
T2_abs = 210
T2_rel = 1.2
T3 = 5
MED_WINDOW_SIZE = 5
GAUSS_BLUR_SIZE = 15
MASK_BLUR_SIZE = 1

__SHOW__ = False


class Method1():
    def run(self, img):
        #
        # Segmentation
        #

        # Step 1 global threshold
        cR, cG, cB = cv2.split(img)
        cE = 0.2989 * cR + 0.5870 * cG + 0.1140 * cB  #  grey scale intensity
        mask_1 = self.get_global_mask(cE, cB, cG, T1)

        # Step 2 Smoothed Nonspecular Surface Colour comparison

        mask_tmp = self.get_global_mask(cE, cB, cG, T2_abs)
        color_map = self.get_filled_map(img, mask_tmp)
        mask_2 = self.get_relative_mask(cR, cG, cB, color_map, T2_rel)

        # combine two masks
        mask = mask_1 + mask_2

        #
        # Inpainting
        #
        smooth_map = cv2.GaussianBlur(color_map,
                                      (GAUSS_BLUR_SIZE, GAUSS_BLUR_SIZE), 0)
        mask = mask.astype(float)
        smooth_mask = cv2.GaussianBlur(mask, (MASK_BLUR_SIZE, MASK_BLUR_SIZE),
                                       0)
        smooth_mask = cv2.merge(
            [smooth_mask.copy(),
             smooth_mask.copy(),
             smooth_mask.copy()])
        res = smooth_mask * smooth_map + (1 - smooth_mask) * img
        res = res.astype(np.uint8)

        if __SHOW__:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(smooth_map)
            plt.subplot(2, 2, 3)
            plt.imshow(smooth_mask)
            plt.subplot(2, 2, 4)
            plt.imshow(res)
            plt.show()

        return res

    def get_global_mask(self, cE, cB, cG, T):

        norm_effe_GE = np.percentile(cG, 95.0) / np.percentile(cE, 95.0)
        norm_effe_BE = np.percentile(cB, 95.0) / np.percentile(cE, 95.0)

        flag_0 = cG > norm_effe_GE * T
        flag_1 = cB > norm_effe_BE * T
        flag_2 = cE > T

        mask_1 = flag_0 + flag_1 + flag_2
        return mask_1

    def get_filled_map(self, img, mask_tmp):
        """fill the masked pixels with surrounded values"""

        # find the centorids
        centroids = self.get_centroid_info(img, mask_tmp)  # [M, 5]

        # prepare the point indexs
        c_points = centroids[:, :2].T[::-1, :]  # centroid [2, M]
        h_points = np.array(np.where(mask_tmp == 1))  # hollow [2, N]
        M = c_points.shape[1]
        N = h_points.shape[1]
        c_points_r = np.repeat([c_points], N, 0)  # [N, 2, M]
        c_points_r = np.swapaxes(c_points_r, 0, 1)  # [2, N, M]
        c_points_r = np.swapaxes(c_points_r, 1, 2)  # [2, M, N]
        h_points_r = np.repeat([h_points], M, 0)  # [M, 2, N]
        h_points_r = np.swapaxes(h_points_r, 0, 1)  # [2, M, N]

        # parallel calculate distance and find nearest color
        dis = np.sum(np.square(h_points_r - c_points_r), axis=0)  # [M, N]
        nearest_centroid_index = np.argmin(dis, axis=0)  # [N]
        hollow_values = centroids[:, 2:][[nearest_centroid_index]]  # [N, 3]

        # fill image with found colors
        img_fill = img.copy()
        h_points = h_points.tolist()
        img_fill[h_points] = hollow_values

        # median blur
        color_map = cv2.medianBlur(img_fill, MED_WINDOW_SIZE)

        return color_map

    def get_centroid_info(self, img, mask_tmp):
        """find the centroid as well as their color of each contour"""

        # find color choose area
        mask_tmp = mask_tmp.astype(np.uint8)
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        delated_2 = cv2.dilate(mask_tmp, kernel_2)
        kernel_4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        delated_4 = cv2.dilate(mask_tmp, kernel_4)
        color_area = delated_4 - delated_2

        # find contours
        contours, __ = cv2.findContours(color_area, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # for each contour
        res = np.zeros([len(contours), 5])
        for i, c in enumerate(contours):
            # get centroid point index
            shape = c.shape
            c = c.reshape([shape[0], shape[2]]).T
            res[i, :2] = centroid_index = np.mean(c, axis=1)

            # get centroid color
            tmp = np.zeros([shape[0], 3])
            c = c[::-1, :].tolist()
            tmp[:, :] = img[c]
            res[i, 2:] = centroid_value = np.mean(tmp, axis=0)
            pass

        return res

    def get_relative_mask(self, cR, cG, cB, color_map, T):
        """
        The second module compares every given
        pixel to a smoothed nonspecular surface colour at the pixel
        position, which is estimated from local image statistics.
        """

        contrast_coeff_R = np.mean(cR) / (np.mean(cR) + np.std(cR))
        contrast_coeff_G = np.mean(cG) / (np.mean(cG) + np.std(cG))
        contrast_coeff_B = np.mean(cB) / (np.mean(cB) + np.std(cB))

        img = cv2.merge([cR, cG, cB])
        intensity_ratio = (img + 1e-10) / (color_map + 1e-10)
        intensity_ratio[:, :, 0] *= contrast_coeff_R
        intensity_ratio[:, :, 1] *= contrast_coeff_G
        intensity_ratio[:, :, 2] *= contrast_coeff_B
        intensity_ratio = np.max(intensity_ratio, axis=2)

        mask_2 = intensity_ratio > T2_rel

        return mask_2


Y_COEFF = 0.5


class Method2():
    def run(self, img):

        #
        # Segmentation
        #

        # image enhance
        img_s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
        img_s = cv2.merge([img_s.copy(), img_s.copy(), img_s.copy()])
        max_s = img_s.max()
        img_enhs = img.copy().astype(float)
        img_enhs = img_enhs * (max_s - img_s) / max_s
        img_enhs = img_enhs.astype(np.uint8)

        # get color_y
        enh_xyz = cv2.cvtColor(img_enhs, cv2.COLOR_RGB2XYZ)
        enh_y = enh_xyz[:, :, 1]
        color_y = (enh_y + 1e-10) / (np.sum(enh_xyz, axis=2) + 1e-10)

        # get mask
        mask = (enh_y / 255) * Y_COEFF > color_y

        #
        # Inpainting
        #
        m1 = Method1()
        color_map = m1.get_filled_map(img, mask)
        smooth_map = cv2.GaussianBlur(color_map,
                                      (GAUSS_BLUR_SIZE, GAUSS_BLUR_SIZE), 0)
        mask = mask.astype(float)
        smooth_mask = cv2.GaussianBlur(mask, (MASK_BLUR_SIZE, MASK_BLUR_SIZE),
                                       0)
        smooth_mask = cv2.merge(
            [smooth_mask.copy(),
             smooth_mask.copy(),
             smooth_mask.copy()])
        res = smooth_mask * smooth_map + (1 - smooth_mask) * img
        res = res.astype(np.uint8)

        # show result
        if __SHOW__:
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(img_enhs)
            plt.subplot(2, 2, 3)
            plt.imshow(smooth_mask)
            plt.subplot(2, 2, 4)
            plt.imshow(res)
            plt.show()

        return res


if __name__ == "__main__":
    import os

    m1 = Method1()
    for file_name in os.listdir('./img/'):
        img = cv2.imread('./img/' + file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = m1.run(img)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./res/' + 'res_1_' + file_name, res)

    m2 = Method2()
    for file_name in os.listdir('./img/'):
        img = cv2.imread('./img/' + file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = m2.run(img)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./res/' + 'res_2_' + file_name, res)

    pass
