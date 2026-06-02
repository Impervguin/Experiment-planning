from typing import Optional
from random import random
from .task import Task
from .exceptions import EventBusyException, EventEmptyException
from distributions import RandomDistribution
from enum import Enum
import heapq



class EventBlock:
    def trigger_input(self, sim, who, task: Task):
        raise NotImplementedError()

    def trigger_output(self, sim, who) -> Task:
        raise NotImplementedError()


class Generator(EventBlock):
    def __init__(self, dist: RandomDistribution, output: EventBlock, name="gen"):
        self.dist = dist
        self.output = output
        self.name = name
        self.generated = 0
        self.inter_arrival_times = []
        self._seq = 0

    def event(self, sim):
        dt = self.dist.generate()
        self.inter_arrival_times.append(dt)
        sim.add_event(self.event, sim.time + dt)

        task = self._generate()
        self._seq += 1
        self.generated += 1
        task.stamp(f"arrival/{self.name}", sim.time)

        try:
            self.output.trigger_input(sim, self, task)
        except EventBusyException:
            pass

    def trigger_input(self, sim, who, task):
        raise EventBusyException()
    
    def _generate(self) -> Task:
        return Task(self._seq)
    
    def average_gen_time(self) -> float:
        return sum(self.inter_arrival_times) / len(self.inter_arrival_times) if len(self.inter_arrival_times) > 0 else 0

class PriorityGenerator(Generator):
    def __init__(self, dist: RandomDistribution, output: EventBlock, name="gen", priority: int = 0):
        super().__init__(dist, output, name)
        self.priority = priority
    
    def _generate(self) -> Task:
        return Task(self._seq, priority=self.priority)


class Queue(EventBlock):
    def __init__(self, output: EventBlock, name="queue"):
        self.output = output
        self.name = name
        self.queue: list[Task] = []

    def trigger_input(self, sim, who, task: Task):
        task.stamp(f"queue_in/{self.name}", sim.time)
        self.queue.append(task)

        try:
            self.output.trigger_input(sim, self, self.queue[0])
            self.queue.pop(0)
        except EventBusyException:
            pass

    def trigger_output(self, sim, who):
        if not self.queue:
            raise EventEmptyException()
        return self.queue.pop(0)


class RelativePriorityQueue(EventBlock):
    """
    Очередь с относительными приоритетами (non-preemptive).
    Меньшее значение priority => более высокий приоритет.
    FIFO внутри одного приоритета.
    """

    def __init__(self, output, name="queue"):
        self.output = output
        self.name = name

        self._heap = []
        self._seq = 0   # для FIFO внутри одного приоритета

    def trigger_input(self, sim, who, task: Task):
        task.stamp(f"queue_in/{self.name}", sim.time)

        # (priority, seq, task)
        heapq.heappush(self._heap, (task.priority, self._seq, task))
        self._seq += 1

        try:
            pr, _, t = self._heap[0]
            self.output.trigger_input(sim, self, t)
            heapq.heappop(self._heap)
        except EventBusyException:
            pass

    def trigger_output(self, sim, who) -> Task:
        if not self._heap:
            raise EventEmptyException()
        _, _, task = heapq.heappop(self._heap)
        return task
    
    def tasks(self) -> list[Task]:
        return [t for _, _, t in self._heap]

class Processor(EventBlock):
    def __init__(
        self,
        dist: RandomDistribution,
        input_block: Queue,
        output: EventBlock,
        name="proc",
        dist_priority_1: RandomDistribution = None
    ):
        self.dist = dist
        self.dist_priority_1 = dist_priority_1
        self.input = input_block
        self.output = output
        self.name = name

        self.processing = False
        self.current: Optional[Task] = None

        self.service_times = []
        self.busy_time = 0.0
        self._last_time = 0.0

    def _get_dist_for_task(self, task: Task) -> RandomDistribution:
        """Get the appropriate distribution based on task priority"""
        if task.priority == 1 and self.dist_priority_1 is not None:
            return self.dist_priority_1
        return self.dist

    def _update_busy(self, sim):
        dt = sim.time - self._last_time
        if self.processing:
            self.busy_time += dt
        self._last_time = sim.time

    def trigger_input(self, sim, who, task: Task):
        if self.processing:
            raise EventBusyException()

        self._update_busy(sim)
        self.processing = True
        self.current = task
        task.stamp(f"start/{self.name}", sim.time)

        dist = self._get_dist_for_task(task)
        dur = dist.generate()
        self.service_times.append(dur)
        sim.add_event(self._finish, sim.time + dur)

    def _finish(self, sim):
        self._update_busy(sim)
        task = self.current
        task.stamp(f"finish/{self.name}", sim.time)

        self.processing = False
        self.current = None

        self.output.trigger_input(sim, self, task)

        try:
            nxt = self.input.trigger_output(sim, self)
            self.trigger_input(sim, self, nxt)
        except EventEmptyException:
            pass

    def trigger_output(self, sim, who):
        raise EventEmptyException()
    
    def average_proc_time(self) -> float:
        return sum(self.service_times) / len(self.service_times) if len(self.service_times) > 0 else 0

class Terminator(EventBlock):
    def __init__(self, name="term"):
        self.name = name
        self.tasks: list[Task] = []

    def trigger_input(self, sim, who, task: Task):
        task.stamp(f"end/{self.name}", sim.time)
        self.tasks.append(task)

    def trigger_output(self, sim, who):
        raise EventEmptyException()