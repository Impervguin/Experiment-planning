import heapq
from typing import Callable


class EventSimulation:
    def __init__(self, stop_condition_fn: Callable[[], bool]):
        self.time = 0.0
        self._queue = []
        self._seq = 0
        self.stop_condition_fn = stop_condition_fn

    def add_event(self, callback: Callable, time: float):
        heapq.heappush(self._queue, (time, self._seq, callback))
        self._seq += 1

    def run(self):
        while self._queue:
            time, _, callback = heapq.heappop(self._queue)
            self.time = time
            callback(self)
            if self.stop_condition_fn():
                break