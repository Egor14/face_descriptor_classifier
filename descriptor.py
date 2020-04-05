import numpy as np
import scipy.fftpack
import cv2
import os


class DescriptorMaker:

    def handle(self, method, params, human, image_number):
        image = self.image_read(human, image_number)
        descriptive_image = getattr(self, method)(params, image)
        return descriptive_image

    def image_read(self, human, image_number):
        image = cv2.imread(os.path.join('orl_faces', 's' + str(human), str(image_number) + '.pgm'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def brightness_hist(self, params, image):
        image = image.reshape((image.shape[0] * image.shape[1],))
        hist = np.histogram(image, params)
        return hist[0]

    def dft(self, params, image):
        image = np.fft.fft2(image, norm='ortho')
        image = np.real(image)
        return self.image_reshape(image[:params, :params])

    def dct(self, params, image):
        image = scipy.fftpack.dct(image, axis=1)
        image = scipy.fftpack.dct(image, axis=0)
        return self.image_reshape(image[:params, :params])

    def scale(self, params, image):
        image = cv2.resize(image, (int(image.shape[1] * params), int(image.shape[0] * params)))
        return self.image_reshape(image)

    def gradient(self, params, image):
        shape = image.shape[0]
        i = 1
        result = []
        while (i) * params + 2 * params <= shape:
            prev = image[i * params:(i) * params + params, :]
            next = image[(i) * params + params:(i) * params + 2 * params, :]
            result.append(prev - next)
            i += 1
        result = np.array(result)
        result = result.reshape((result.shape[0] * result.shape[1], result.shape[2]))
        result = np.mean(result, axis=0)
        return result

    def image_reshape(self, result):
        return result.reshape((result.shape[0] * result.shape[1],))
