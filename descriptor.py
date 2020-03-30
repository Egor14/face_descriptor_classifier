import numpy as np
import scipy.fftpack
# import numpy.
import math
import cv2
import os
from matplotlib import pyplot as plt


class DescriptorMaker:

    def handle(self, method, params, human, image_number):
        image = self.image_read(human, image_number)
        descriptive_image = getattr(self, method)(params, image)
        return descriptive_image

    def image_read(self, human, image_number):
        image = cv2.imread(os.path.join('orl_faces', 's' + str(human), str(image_number) + '.pgm'), 0)
        return image

    def brightness_hist(self, params, image):
        image = image.reshape((image.shape[0] * image.shape[1],))
        color_range = range(256)
        interval_len = math.ceil(len(color_range) / params)
        hist = [0] * params
        for i in image:
            hist[i//interval_len] += 1

        return hist

    def dft(self, params, image):
        print(image.shape)
        image = np.fft.fft2(image, norm='ortho')
        # print(image)
        print(image.shape)
        plt.imshow(np.real(image), cmap='gray')
        print(np.real(image)[:5,:5])
        print(image[:5,:5])
        plt.show()

    def dct(self, params, image):
        image = scipy.fftpack.dct(image, axis=1)
        image = scipy.fftpack.dct(image, axis=0)
        print(image)
        print(image.shape)
        plt.imshow(image[:5, :5], cmap='gray')
        plt.show()

    def scale(self, params, image):
        result = cv2.resize(image, (int(image.shape[1] * params), int(image.shape[0] * params)))
        result = result.reshape((result.shape[0] * result.shape[1],))
        return result

    def gradient(self, params, image):
        pass


if __name__ == '__main__':
    duc = DescriptorMaker()
    duc.brightness_hist(1, 1)
