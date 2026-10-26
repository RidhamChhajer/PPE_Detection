import time

import numpy as np
import pytest

from app.camera_driver import DriverCapture
from app.cameras import CameraPool
from app.config import Settings, ViewConfig
from tests.test_core import wait_until


def blocked_driver(index, connection, memory_name):
    while True:
        operation, args = connection.recv()
        if operation == "isOpened":
            connection.send((True, True))
        elif operation in {"read", "release"}:
            time.sleep(60)
        else:
            connection.send((True, 0))


def frame_driver(index, connection, memory_name):
    from multiprocessing.shared_memory import SharedMemory

    memory = SharedMemory(name=memory_name)
    value = 20
    try:
        while True:
            operation, args = connection.recv()
            if operation == "release":
                connection.send((True, None))
                return
            if operation == "read":
                np.ndarray((4, 5, 3), dtype=np.uint8, buffer=memory.buf)[:] = value
                value += 1
                result = (True, (4, 5, 3))
            else:
                result = True
            connection.send((True, result))
    finally:
        memory.close()


def test_blocked_native_read_times_out_and_worker_can_be_terminated():
    capture = DriverCapture(0, worker=blocked_driver, timeout=0.2)
    try:
        assert capture.isOpened()
        capture.has_frame = True
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out during read"):
            capture.read()
        capture.release()
        assert time.monotonic() - start < 3
        assert capture.closed
    finally:
        capture.release()


def test_shared_frames_are_owned_copies_and_release_is_repeatable():
    for _ in range(2):
        capture = DriverCapture(0, worker=frame_driver)
        try:
            assert capture.isOpened()
            ok, first = capture.read()
            assert ok and first.shape == (4, 5, 3)
            ok, second = capture.read()
            assert ok and np.all(second == 21)
            assert np.all(first == 20)
        finally:
            capture.release()
            capture.release()


def test_camera_pool_can_stop_and_restart_even_when_native_read_never_returns():
    captures = []

    def factory(index):
        capture = DriverCapture(index, worker=blocked_driver)
        captures.append(capture)
        return capture

    pool = CameraPool(factory)
    settings = Settings(upper=ViewConfig(camera=0), lower=ViewConfig(camera=None))
    for _ in range(2):
        pool.start(settings)
        try:
            wait_until(lambda: pool.devices[0].actual.get("requested_fps") == 30, seconds=5)
        finally:
            start = time.monotonic()
            assert pool.stop()
            assert time.monotonic() - start < 3
        assert pool.devices == {}
        assert captures[-1].closed
