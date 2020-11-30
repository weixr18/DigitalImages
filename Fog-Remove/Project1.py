import cv2
import numpy as np


class DarkChannel():
    def zmMinFilterGray(self, src, r=7):
        '''最小值滤波，r是滤波器半径'''
        return cv2.erode(src, np.ones((2 * r - 1, 2 * r - 1)))

    def guidedfilter(self, I, p, r, eps):
        '''引导滤波'''
        height, width = I.shape
        m_I = cv2.boxFilter(I, -1, (r, r))
        m_p = cv2.boxFilter(p, -1, (r, r))
        m_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p

        m_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = m_II - m_I * m_I

        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I

        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b

    def getV1(self, m, r, eps, w, maxV1):  #输入rgb图像，值范围[0,1]
        '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        # 得到暗通道图像
        V1 = np.min(m, 2)
        # 使用引导滤波优化
        V1 = self.guidedfilter(V1, self.zmMinFilterGray(V1, 7), r, eps)
        bins = 2000
        ht = np.histogram(V1, bins)  #计算大气光照A
        d = np.cumsum(ht[0]) / float(V1.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
        V1 = np.minimum(V1 * w, maxV1)  #对值范围进行限制
        return V1, A

    def deHaze(self, m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
        Y = np.zeros(m.shape)
        V1, A = self.getV1(m, r, eps, w, maxV1)  #得到遮罩图像和大气光照
        for k in range(3):
            Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  #颜色校正
        Y = np.clip(Y, 0, 1)
        if bGamma:
            Y = Y**(np.log(0.5) / np.log(Y.mean()))  #gamma校正,默认不进行该操作
        return Y

    def run(self, img):
        img_n = img / 255.0
        img_n_dehaze = self.deHaze(img_n)
        img_dehaze = img_n_dehaze * 255
        return img_dehaze


class CLAHE():
    def clahe(self, image, denoise=True, verbose=False, limit=2):
        bgr = image

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=limit)
        lab_planes[0] = clahe.apply(lab_planes[0])
        #lab_planes[1] = clahe.apply(lab_planes[1])
        #lab_planes[2] = clahe.apply(lab_planes[2])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if denoise:
            bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
            #bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

        if verbose:
            cv2.imshow("test", bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return bgr

    def run(self, img):
        return self.clahe(img)


if __name__ == '__main__':
    import os

    dark_channel = DarkChannel()
    clahe = CLAHE()

    for file_name in os.listdir('./img/'):
        img = cv2.imread('./img/' + file_name)
        img_dehaze = dark_channel.run(img)
        cv2.imwrite('./res/' + 'dark_' + file_name, img_dehaze)

    for file_name in os.listdir('./img/'):
        img = cv2.imread('./img/' + file_name)
        img_dehaze = clahe.run(img)
        cv2.imwrite('./res/' + 'clahe_' + file_name, img_dehaze)
    pass