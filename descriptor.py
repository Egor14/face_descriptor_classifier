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
            hist[i // interval_len] += 1

        return np.array(hist)

    def dft(self, params, image):
        image = np.fft.fft2(image, norm='ortho')
        return self.image_reshape(image[:params, :params])

    def dct(self, params, image):
        image = scipy.fftpack.dct(image, axis=1)
        image = scipy.fftpack.dct(image, axis=0)
        return self.image_reshape(image[:params, :params])

    def scale(self, params, image):
        image = cv2.resize(image, (int(image.shape[1] * params), int(image.shape[0] * params)))
        # plt.imshow(image)
        # plt.show()
        return self.image_reshape(image)

    def gradient(self, params, image):
        image = np.gradient(image, axis=0)
        image = np.gradient(image, axis=1)
        return self.image_reshape(image[:params, :params])

    def image_reshape(self, result):
        return result.reshape((result.shape[0] * result.shape[1],))
