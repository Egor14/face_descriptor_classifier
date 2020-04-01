import random
import sys
import csv
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from decimal import Decimal

from descriptor import DescriptorMaker


class FaceFinder:

    def __init__(self):
        self.methods = ['brightness_hist', 'dft', 'dct', 'scale', 'gradient']
        self.descriptor_maker = DescriptorMaker()
        if not all([os.path.exists(method + '.csv') for method in self.methods]):
            self.make_all_descriptors()

    def make_all_descriptors(self):
        for method in self.methods:
            with open(method + '.csv', 'w') as file:
                writer = csv.DictWriter(file, fieldnames=['value', 'target'])
                writer.writeheader()
                for human in range(1, 41):
                    for image_number in range(1, 11):
                        result = getattr(self.descriptor_maker, method)(human, image_number)
                        writer.writerow({'value': result, 'target': human})

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
                    'target': human
                })
                # break
            # break
        return data

    def cross_val(self, method, params):
        accuracies = []
        for test_size in range(1, 11):
            for param in params:
                cross = 0
                for _ in range(5):
                    cross += self.find_face(method, param, test_size)
                accuracies.append((cross / 5, param, test_size))

        print('CROSS VAL BEST ESTIMATOR', max(accuracies, key=lambda i: i[0]))

    def train_test_split(self, data, size):
        test_indexes = random.sample(range(1, 11), k=size)
        # test_indexes = [5]
        train_data = [i for i in data if i['id'] % 10 not in test_indexes]
        # train_data = data
        return train_data, test_indexes

    def worker(self, method, params, train_data, human, image_number):
        new_image = self.descriptor_maker.handle(method, params, human, image_number)
        detection_results = []
        mean = 0
        for i, face in enumerate(train_data):
            #     if (i + 1) % 10 == 0:
            #         detection_results.append(mean / 10)
            #         mean = 0
            #     mean += self.distance(face['value'], new_image)

            detection_results.append(self.distance(face['value'], new_image))

        # photo = np.array(detection_results).argmin() + 1
        # plt.imshow(self.descriptor_maker.image_read(photo//10, photo%10))
        # plt.show()
        # print(len(detection_results))

        # print(np.array(detection_results).argmin() + 1)
        answer = math.ceil((np.array(detection_results).argmin() + 1) / (len(train_data) / 40))
        # answer = np.array(detection_results).argmin() + 1
        # print(len(detection_results))
        return answer

    def distance(self, x, y):
        return np.sum(np.abs(x - y))

    def find_face(self, method, params, test_size):
        prepared_data = self.make_descriptors(method, params)
        train_data, test_indexes = self.train_test_split(prepared_data, test_size)
        count = 0
        predicted = []
        for human in range(1, 41):
            for image_number in test_indexes:
                answer = self.worker(method, params, train_data, human, image_number)
                # print(answer, human)
                predicted.append(int(answer == human))
                count += 1

        accuracy = sum(predicted) / count
        print(round(accuracy, 3), 'param: ', params, 'test_size: ', test_size)
        return accuracy
        # break
        # break


if __name__ == '__main__':
    finder = FaceFinder()
    # finder.find_face('scale', 0.5, 1)
    # finder.find_face('brightness_hist', 20, 1)
    # finder.find_face('scale', 0.7, 1)
    finder.cross_val('scale', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
