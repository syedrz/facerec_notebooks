from abc import ABCMeta, abstractmethod
from .utils import get_logger
import cv2
import time
import asyncio

logger = get_logger(__name__)


class UnreadableSource(Exception):
    """Unreadable source."""


class UnopenedSource(Exception):
    """Unopened source."""


class Source(metaclass=ABCMeta):

    @abstractmethod
    async def __aiter__(self):
        yield NotImplementedError

    @abstractmethod
    async def __anext__(self):
        yield NotImplementedError


class LiveSource(Source):

    def __init__(self, source=0):
        self._source = source
        self.src = cv2.VideoCapture(source)
        self.res = (int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.fps = self.src.get(cv2.CAP_PROP_FPS) or 30.0

    async def __aiter__(self):
        while self.src.isOpened():
            yield self._read()
            await asyncio.sleep(1 / self.fps)
        raise UnopenedSource()

    async def __anext__(self):
        if self.src.isOpened():
            yield self._read()
        else:
            raise UnopenedSource()

    def _read(self):
        ret, frame = self.src.read()
        if ret:
            return frame, round(time.time() * 1000)
        else:
            raise UnreadableSource()

    def release(self):
        self.src.release()

    def __repr__(self):
        return f'<{self.__class__.__name__} {self._source} @ {self.res}>'


class StaticSource(LiveSource):

    def __init__(self, source):
        super().__init__(source=source)
        self.start = time.time()

    def _read(self):
        timestamp = round((time.time() - self.start) * 1000)
        self.src.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        ret, frame = self.src.read()
        if ret:
            return frame, timestamp
        else:
            raise UnreadableSource()
