from .utils import get_logger
from abc import ABCMeta, abstractmethod
import cv2


class Detector(metaclass=ABCMeta):

    @abstractmethod
    async def _detect(self, image):
        yield NotImplementedError

    async def __call__(self, image):
        async for i, x, y, w, h in self._detect(image):
            yield image[y:y + h, x:x + w], i, x, y, w, h


class CascadeDetector(Detector):

    def __init__(self, face_cascade):
        self.face_cascade = cv2.CascadeClassifier(face_cascade)

    def _detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        for i, (x, y, w, h) in enumerate(faces):
            yield i, x, y, w, h
