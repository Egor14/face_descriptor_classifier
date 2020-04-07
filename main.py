import random
import csv
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from settings_local import *

from descriptor import DescriptorMaker


class FaceFinder:

    def __init__(self):
        self.methods = ['brightness_hist', 'dft', 'dct', 'scale', 'gradient']
        self.descriptor_maker = DescriptorMaker()

    def make_descriptors(self, method, params):
        count = 0
        data = []
        for human in range(1, 41):
            for image_number in range(1, 11):
                count += 1
                result = self.descriptor_maker.handle(method, params, human, image_number)
                data.append({
                    'id': count,
                    'value': result,
                    'target': human,
                    'image_number': image_number
                })

        return data

    def cross_val(self, method, params):
        accuracies = []
        for test_size in range(1, 10):
            for param in params:
                cross = 0
                for _ in range(5):
                    val, predicted = self.find_face(method, param, test_size)
                    cross += val
                accuracies.append((cross / 5, param, test_size))

        accuracies = np.array(accuracies)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.array(accuracies[:, 1]), np.array(accuracies[:, 2]), np.array(accuracies[:, 0]), c='r',
                   marker='o')
        ax.set_xlabel('Params')
        ax.set_ylabel('Test size')
        ax.set_zlabel('Accuracy')
        plt.show()

        print('CROSS VAL BEST ESTIMATOR', max(accuracies, key=lambda i: i[0]))

    def voting_classifier(self):
        voting_accuracies = []
        for method in self.methods:
            voting_predicted = []
            accuracies = []
            for test_size in range(1, 10):
                cross = 0
                predicted = 0
                for _ in range(5):
                    val, predicted = self.find_face(method, best_params[method], test_size)
                    cross += val
                accuracies.append(cross / 5)
                voting_predicted.append(predicted)
            voting_accuracies.append(voting_predicted)
            plt.plot(range(1, 10), list(reversed(accuracies)), label=method)

        accuracies = []
        voting_accuracies = np.array(voting_accuracies)
        for col in range(voting_accuracies.shape[1]):
            ls_len = len(voting_accuracies[:, col][0])
            res = np.zeros(ls_len)
            for ls in voting_accuracies[:, col]:
                res += np.array(ls)
            count = 0
            for i in res:
                if i >= 3:
                    count += 1
            accuracies.append(count / res.shape[0])
        plt.plot(range(1, 10), list(reversed(accuracies)), label='voting')

        plt.legend(title='Methods:')
        plt.xlabel("Train size")
        plt.ylabel("Accuracy")
        plt.show()

    def train_test_split(self, data, size):
        test_indexes = random.sample(range(1, 11), k=size)
        train_data = [i for i in data if i['id'] % 10 not in test_indexes]
        if 10 in test_indexes:
            train_data = [i for i in train_data if i['id'] % 10 != 0]

        return train_data, test_indexes

    def worker(self, method, params, train_data, human, image_number, show_images):
        new_image = self.descriptor_maker.handle(method, params, human, image_number)
        detection_results = []
        for i, face in enumerate(train_data):
            dist = self.distance(face['value'], new_image)

            detection_results.append(dist)

        detected_point_ind = np.array(detection_results).argmin()
        found_image = self.descriptor_maker.image_read(train_data[detected_point_ind]['target'],
                                                       train_data[detected_point_ind]['image_number'])
        my_image = self.descriptor_maker.image_read(human, image_number)
        descriptive_my_image = new_image
        descriptive_found_image = train_data[detected_point_ind]['value']

        if show_images:
            self.show_images(method, my_image, found_image, descriptive_my_image, descriptive_found_image, params)

        answer = math.ceil((detected_point_ind + 1) / (len(train_data) / 40))
        return answer

    @staticmethod
    def close_event():
        plt.close()

    def show_images(self, method, my_image, found_image, descriptive_my_image, descriptive_found_image, params):
        plt.subplot(221)
        plt.imshow(my_image, cmap='gray')
        plt.title('My image')

        plt.subplot(222)
        plt.imshow(found_image, cmap='gray')
        plt.title('Detected Point')

        if method not in ['brightness_hist', 'gradient']:
            if method == 'scale':
                params = reshapes[str(params)]
            else:
                params = (params, params)
            descriptive_my_image = descriptive_my_image.reshape(params)
            descriptive_found_image = descriptive_found_image.reshape(params)

            plt.subplot(223)
            plt.imshow(descriptive_my_image, cmap='gray')
            plt.title('My image descriptor')

            plt.subplot(224)
            plt.imshow(descriptive_found_image, cmap='gray')
            plt.title('Detected Point descriptor')
        else:
            plt.subplot(223)
            plt.hist(descriptive_my_image)
            plt.title('My image descriptor')

            plt.subplot(224)
            plt.hist(descriptive_found_image)
            plt.title('Detected Point descriptor')

        plt.show()

    def distance(self, x, y):
        return np.sum((x - y) ** 2)

    def find_face(self, method, params, test_size, show_images=False):
        prepared_data = self.make_descriptors(method, params)
        train_data, test_indexes = self.train_test_split(prepared_data, test_size)
        count = 0
        predicted = []
        for human in range(1, 41):
            for image_number in test_indexes:
                answer = self.worker(method, params, train_data, human, image_number, show_images)
                predicted.append(int(answer == human))
                count += 1

        accuracy = sum(predicted) / count
        print(round(accuracy, 3), 'param: ', params, 'test_size: ', test_size)
        return accuracy, predicted


if __name__ == '__main__':
    finder = FaceFinder()
    # finder.find_face('gradient', 6, 1, show_images=True)
    # finder.cross_val('gradient', [2, 4, 6, 8, 10])
    finder.voting_classifier()
