"""Keep blocking Windows camera calls outside the application process."""

import multiprocessing
import os
import threading
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np


def _exit_with_parent():
    parent = multiprocessing.parent_process()
    if parent is not None:
        parent.join()
        os._exit(0)


def _camera_host(index, connection, memory_name):
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    memory = SharedMemory(name=memory_name)
    capture = None
    try:
        if isinstance(index, str):
            from cv2_enumerate_cameras import enumerate_cameras

            device = next((c for c in enumerate_cameras(cv2.CAP_DSHOW) if c.path == index), None)
            if device is None:
                raise RuntimeError("Selected camera was disconnected before opening.")
            index = device.index
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        while True:
            operation, args = connection.recv()
            if operation == "read":
                ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    result = (False, None)
                else:
                    if frame.dtype != np.uint8 or frame.nbytes > memory.size:
                        raise RuntimeError("Camera frame exceeds the supported 4K image format.")
                    np.ndarray(frame.shape, dtype=np.uint8, buffer=memory.buf)[:] = frame
                    result = (True, frame.shape)
            elif operation == "release":
                capture.release()
                connection.send((True, None))
                break
            elif operation in {"isOpened", "get", "set"}:
                result = getattr(capture, operation)(*args)
            else:
                raise RuntimeError("Unsupported camera operation")
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send((False, str(exc)))
        except (OSError, EOFError):
            pass
    finally:
        if capture is not None:
            capture.release()
        connection.close()
        memory.close()


class DriverCapture:
    """OpenCV-compatible capture with bounded calls and an interruptible owner."""

    def __init__(self, index, worker=_camera_host, timeout=3):
        context = multiprocessing.get_context("spawn")
        self.memory = SharedMemory(create=True, size=3840 * 2160 * 4)
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=worker, args=(index, child, self.memory.name), daemon=True
        )
        self.timeout = timeout
        self.closed = False
        self.has_frame = False
        try:
            self.process.start()
        except Exception:
            self.connection.close()
            self.memory.close()
            self.memory.unlink()
            raise
        finally:
            child.close()

    def _call(self, operation, *args, timeout=None):
        if self.closed:
            raise RuntimeError("Camera is closed")
        try:
            self.connection.send((operation, args))
            if not self.connection.poll(self.timeout if timeout is None else timeout):
                raise RuntimeError(f"Camera driver timed out during {operation}. Reopening camera.")
            ok, result = self.connection.recv()
        except (OSError, EOFError) as exc:
            raise RuntimeError("Camera driver stopped responding. Reopening camera.") from exc
        if not ok:
            raise RuntimeError(result)
        return result

    def isOpened(self):
        return self._call("isOpened", timeout=8)

    def get(self, key):
        return self._call("get", key)

    def set(self, key, value):
        return self._call("set", key, value)

    def read(self):
        ok, shape = self._call("read", timeout=self.timeout if self.has_frame else 8)
        if not ok:
            return False, None
        self.has_frame = True
        return True, np.ndarray(shape, dtype=np.uint8, buffer=self.memory.buf).copy()

    def interrupt(self):
        if not self.closed and self.process.is_alive():
            self.process.terminate()

    def release(self):
        if self.closed:
            return
        try:
            if self.process.is_alive():
                try:
                    self._call("release", timeout=1.5)
                    self.process.join(timeout=0.5)
                except RuntimeError:
                    pass
                self.interrupt()
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=1)
        finally:
            self.closed = True
            self.connection.close()
            self.memory.close()
            self.memory.unlink()
            if not self.process.is_alive():
                self.process.close()
