
import multiprocessing as mp
import ctypes
import numpy as np

from rlpyt.utils.buffer import np_mp_array
from rlpyt.utils.synchronize import RWLock


class AsyncReplayBufferMixin:
    """Mixin class which manages the buffer (shared) memory under a read-write
    lock (multiple-reader, single-writer), for use with the asynchronous
    runner. Wraps the ``append_samples()``, ``sample_batch()``, and
    ``update_batch_priorities()`` methods. Maintains a universal buffer
    cursor, communicated asynchronously.  Supports multiple buffer-writer
    processes and multiple replay processes.
    """

    async_ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_t = np_mp_array(1, np.uint8)  # Type c_long.
        self.rw_lock = RWLock()
        self._async_buffer_full = np_mp_array(1, np.bool)

    def append_samples(self, *args, **kwargs):
        with self.rw_lock.write_lock:
            self._async_pull()  # Updates from other writers.
            ret = super().append_samples(*args, **kwargs)
            self._async_push()  # Updates to other writers + readers.
        return ret

    def sample_batch(self, *args, **kwargs):
        with self.rw_lock:  # Read lock.
            self._async_pull()  # Updates from writers.
            return super().sample_batch(*args, **kwargs)

    def update_batch_priorities(self, *args, **kwargs):
        with self.rw_lock.write_lock:
            return super().update_batch_priorities(*args, **kwargs)

    def _async_pull(self):
        self.t = self.async_t[0]
        self._buffer_full = self._async_buffer_full[0]

    def _async_push(self):
        self.async_t[0] = self.t
        self._async_buffer_full[0] = self._buffer_full
