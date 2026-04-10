from eventcore import EventSimulation
from eventcore import (
    Generator,
    PriorityGenerator,
    Processor,
    Terminator,
)
from eventcore import Task
from eventcore import RelativePriorityQueue


class SingleServerSMO:
    """
    Одноканальная разомкнутая СМО
    2 генератора -> очередь с относительными приоритетами -> сервер -> выход
    """

    def __init__(
        self,
        gen1_dist,
        gen2_dist,
        service_dist,
        stop_task_count: int,
        service_dist_priority_1=None
    ):
        self.stop_task_count = stop_task_count
        self.done = False

        # --- терминатор ---
        self.terminator = Terminator("sink")

        # --- процессор ---
        self.processor = Processor(
            dist=service_dist,
            input_block=None,          
            output=self.terminator,
            name="server",
            dist_priority_1=service_dist_priority_1
        )

        # --- очередь с относительными приоритетами ---
        self.queue = RelativePriorityQueue(
            output=self.processor,
            name="queue"
        )
        self.processor.input = self.queue

        # --- генераторы ---
        self.gen1 = PriorityGenerator(
            dist=gen1_dist,
            output=self.queue,
            name="gen1",
            priority=0
        )

        self.gen2 = PriorityGenerator(
            dist=gen2_dist,
            output=self.queue,
            name="gen2",
            priority=1
        )

        # --- симуляция ---
        self.sim = EventSimulation(
            stop_condition_fn=lambda: len(self.processor.service_times) >= self.stop_task_count
        )

    # ---------- запуск ----------
    def run(self):
        # старт генераторов
        self.sim.add_event(self.gen1.event, 0.0)
        self.sim.add_event(self.gen2.event, 0.1)
        self.sim.run()
        self.done = True
    
    def theory_util(self) -> float:
        """
        Теоретическая загрузка
        """
        return (self.gen1.dist.intensity() + self.gen2.dist.intensity()) / self.processor.dist.intensity()
    
    def fact_util(self) -> float:
        """
        Фактическая загрузка
        """
        if not self.done:
            return 0
        return (1/self.gen1.average_gen_time() + 1/self.gen2.average_gen_time()) * self.processor.average_proc_time()
    
    def avg_waiting_time(self) -> float:
        """
        Среднее время ожидания в очереди
        """
        waits = []
        for task in self.terminator.tasks:
            if "queue_in/queue" in task.times and "start/server" in task.times:
                waits.append(
                    task.times["start/server"] - task.times["queue_in/queue"]
                )
        for task in self.queue.tasks():
            if task.priority != prio:
                continue
            if "queue_in/queue" in task.times:
                waits.append(
                    self.sim.time - task.times["queue_in/queue"]
                )
        return sum(waits) / len(waits) if waits else 0.0
    
    def avg_waiting_time_priority(self, prio: int) -> float:
        """
        Среднее время ожидания в очереди с относительными приоритетами
        """
        waits = []
        for task in self.terminator.tasks:
            if task.priority != prio:
                continue
            if "queue_in/queue" in task.times and "start/server" in task.times:
                waits.append(
                    task.times["start/server"] - task.times["queue_in/queue"]
                )
        for task in self.queue.tasks():
            if task.priority != prio:
                continue
            if "queue_in/queue" in task.times:
                waits.append(
                    self.sim.time - task.times["queue_in/queue"]
                )

        return sum(waits) / len(waits) if waits else 0.0

    def avg_system_time(self) -> float:
        """
        Среднее время пребывания в системе
        """
        times = []
        for task in self.terminator.tasks:
            arrival_time = None
            if "arrival/gen1" in task.times:
                arrival_time = task.times["arrival/gen1"]
            elif "arrival/gen2" in task.times:
                arrival_time = task.times["arrival/gen2"]

            if arrival_time and "end/sink" in task.times:
                times.append(task.times["end/sink"] - arrival_time)
        return sum(times) / len(times) if times else 0.0
