from abc import ABCMeta, abstractmethod
import cv2
from glob import glob
import os


class DataGen(metaclass=ABCMeta):

    @abstractmethod
    def __iter__(self):
        yield NotImplementedError


class TestDataGen(DataGen):

    def __init__(self, path='data', numeric_y=False):
        self.iter = iter(glob(os.path.join(path, '*', '*', '*_face.jpg')))
        self.numeric_y = numeric_y
        self.map = class_map(path)

    def __iter__(self):
        for path in self.iter:
            _, y, _, _ = path.rsplit(os.path.sep, 4)
            if self.numeric_y:
                y = self.map[y]
            yield cv2.imread(path), y

    def __next__(self):
        path = next(self.iter)
        _, y, _, _ = path.rsplit(os.path.sep, 4)
        if self.numeric_y:
            y = self.map[y]
        return cv2.imread(path), y


def class_map(path='data'):
    return {p.split(os.path.sep)[-1]: i
            for i, p in enumerate(glob(os.path.join(path, '*')))}
