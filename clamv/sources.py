from abc import ABCMeta, abstractmethod
from typing import Iterable, Optional, Tuple
import numpy as np
import cv2
import time

__all__ = ['LiveSource', 'StaticSource']

Frame = Optional[np.ndarray]


class Source(metaclass=ABCMeta):
    """
    Video source base class, designed as an iterable.

    See Also
    --------
    LiveSource : Live video source
    StaticSource : Static video source
    """

    @abstractmethod
    def __iter__(self) -> Iterable[Tuple[Frame, int]]:
        """
        Frame-by-frame iterator over video source.

        Returns
        -------
        frame : np.ndarray or None
            Raw RGB pixel values of video frame
        timestamp : int
            Milisecond timestamp
        """
        yield NotImplementedError

    @abstractmethod
    def __next__(self) -> Tuple[Frame, int]:
        """
        Get a single current frame from video source.

        Returns
        -------
        frame : np.ndarray or None
            Raw RGB pixel values of video frame
        timestamp : int
            Milisecond timestamp
        """
        yield NotImplementedError


class LiveSource(Source):
    """
    Live video source, allows iteration over frames of a live video.

    Parameters
    ----------
    source : int, optional
        Live device id
    fps : float, optional
        Default device capture framerate
    force_fps : bool, optional
        Forcibly override device reported capture framerate

    See Also
    --------
    StaticSource : Static video source
    """

    def __init__(self, source: int = 0, fps: float = 30.0,
                 force_fps: bool = False) -> None:
        self._source = source
        self.src = cv2.VideoCapture(source)
        self.res = (int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if force_fps or self.src.get(cv2.CAP_PROP_FPS) == 0:
            self.fps = fps
        else:
            self.fps = self.src.get(cv2.CAP_PROP_FPS)
        self.last_read = time.time()
        self.open = True

    def __iter__(self) -> Iterable[Tuple[Frame, int]]:
        """
        Frame-by-frame iterator over video source.

        Returns
        -------
        frame : np.ndarray or None
            Raw RGB pixel values of video frame
        timestamp : int
            Milisecond timestamp
        """
        while self.src.isOpened():
            yield self._read()
            read_time = self.last_read + 1 / self.fps
            delta = max(read_time - time.time(), 1e-3)
            time.sleep(delta)
            self.last_read = read_time
        if self.open:
            self.open = False
            yield self._read()
        else:
            raise StopIteration()

    def __next__(self) -> Tuple[Frame, int]:
        """
        Get a single current frame from video source.

        Returns
        -------
        frame : np.ndarray or None
            Raw RGB pixel values of video frame
        timestamp : int
            Milisecond timestamp
        """
        if self.src.isOpened():
            yield self._read()
        elif self.open:
            self.open = False
            yield self._read()
        else:
            raise StopIteration()

    def _read(self) -> Tuple[Frame, int]:
        """
        Get a single frame from video camera with timestamp.

        Returns
        -------
        frame : np.ndarray or None
            Raw RGB pixel values of video frame
        timestamp : int
            Milisecond timestamp
        """
        _, frame = self.src.read()
        return frame, round(time.time() * 1000)

    def release(self) -> None:
        """
        Release control of source device
        """
        self.src.release()


class StaticSource(LiveSource):
    """
    Static source base class, designed as an iterable.

    Parameters
    ----------
    source : str
        Path to static file to be used in place of live video
    fps : float, optional
        Default device capture framerate
    force_fps : bool, optional
        Forcibly override device reported capture framerate

    See Also
    --------
    LiveSource : Live video source
    """

    def __init__(self, source: str, fps: float = 30.0,
                 force_fps: bool = False) -> None:
        super().__init__(source=source)
        self.start = time.time()

    def _read(self) -> Tuple[Frame, int]:
        """
        Get a single frame from video camera with timestamp.

        Returns
        -------
        frame : np.ndarray or None
            Raw RGB pixel values of video frame
        timestamp : int
            Milisecond timestamp
        """
        timestamp = round((time.time() - self.start) * 1000)
        self.src.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        _, frame = self.src.read()
        return frame, timestamp
