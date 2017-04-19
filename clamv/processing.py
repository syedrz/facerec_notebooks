from abc import ABCMeta, abstractmethod
import cv2
import numpy as np


class Processor(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, image):
        return NotImplementedError


class Resizer(Processor):

    def __init__(self, width, height, interpolation=cv2.INTER_LANCZOS4,
                 flatten=False):
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self.flatten = flatten

    def __call__(self, image):
        image = cv2.resize(image, (self.width, self.height),
                           interpolation=self.interpolation)
        if self.flatten:
            return image.flatten()
        return image


class FormatConverter(Processor):

    def __init__(self, source, dest, flip_axis=False):
        self.format = cv2.__dict__[f'COLOR_{source}2{dest}']
        self.flip_axis = flip_axis

    def __call__(self, image):
        converted = cv2.cvtColor(image, self.format)
        if self.flip_axis:
            for i in range((len(converted.shape) - 1) // 2):
                converted = np.swapaxes(converted, i, -i - 1)
        return converted
