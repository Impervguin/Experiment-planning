class Task:
    def __init__(self, task_id: int, priority: int = 0, task_type: int = 0):
        self.id = task_id
        self.priority = priority   # меньше = выше приоритет
        self.type = task_type
        self.times: dict[str, float] = {}

    def stamp(self, key: str, time: float):
        self.times[key] = time