from math import sqrt, pi, log
from random import random

class RandomDistribution:
    # 1/M[x]
    def intensity(self) -> float:
        raise NotImplementedError  

    def generate(self) -> float:
        raise NotImplementedError


class ExponentialDistribution(RandomDistribution):
    def __init__(self, lambda_: float):
        self._lambda = lambda_

    def intensity(self) -> float:
        return 1/self._lambda

    def generate(self) -> float:
        return -self._lambda * log(random())


class RayleighDistribution(RandomDistribution):
    def __init__(self, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("Параметр sigma должен быть положительным")
        self._sigma = sigma
    
    def intensity(self) -> float:
        return sqrt(2 / (pi)) / self._sigma

    def generate(self) -> float:
        u = random()
        return self._sigma * sqrt(-2 * log(u))