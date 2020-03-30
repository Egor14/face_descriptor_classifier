import random
import sys
import csv
import os

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
                break
            # break
        return data

    def cross_val(self):
        pass

    def train_test_split(self, data, size):
        train_indexes = random.sample(range(1, 11), k=size)
        train_indexes = [i ]
        train_data = [i for i in data if i['id'] in train_indexes]
        test_data = [i for i in data if i in train_indexes]

    def find_face(self, method, params, test_size):
        prepared_data = self.make_descriptors(method, params)
        print(prepared_data)


if __name__ == '__main__':
    finder = FaceFinder()
    # finder.find_face('scale', 0.5, 1)
    # finder.find_face('brightness_hist', 20, 1)
    finder.find_face('dft', 20, 1)
