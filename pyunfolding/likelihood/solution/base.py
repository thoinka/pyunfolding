import numpy as np


ONE_SIGMA = 0.682689492137085897

class SolutionBase:
    def __init__(self, likelihood, *args, **kwargs):
        self.likelihood = likelihood

    def solve(self, *args, **kwargs):
        raise NotImplementedError("Solve routine needs to be implemented.")