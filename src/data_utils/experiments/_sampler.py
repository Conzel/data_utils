from typing import Literal
import numpy as np


class AdaptiveSampler:
    """Adaptive sampler for probing a functional with respect to a parameter.
    If the sampler finishes before the maximum number of iterations (default: 100), it is guaranteed
    that the function values are within the maximum step defined in the constructor.
    """

    def __init__(
        self,
        val_range: tuple[float, float],
        max_function_step: float,
        max_iter=100,
        initial_sampling: Literal["linear", "log"] = "log",
        initial_sampling_points=10,
    ):
        a0, a1 = val_range
        if initial_sampling == "log" and a0 <= 0:
            raise ValueError("Logarithmic sampling requires positive range.")

        if initial_sampling == "log":
            self.a_stack = np.logspace(
                np.log10(a0), np.log10(a1), initial_sampling_points
            ).tolist()
        else:
            self.a_stack = np.linspace(a0, a1, initial_sampling_points).tolist()
        self.vals = []
        self.a = []
        self.max_function_step = max_function_step
        self.i = 0
        self.max_iter = max_iter

    def manually_add(self, a: float):
        self.a_stack.append(a)

    def get_next(self):
        self.i += 1
        if self.i > self.max_iter:
            return None
        if len(self.a) != len(self.vals):
            raise ValueError(
                "Values and parameters not in sync. Did you forget to record a value?"
            )
        if len(self.a_stack) == 0:
            self.add_new_a()
            if len(self.a_stack) == 0:
                return None
        a = self.a_stack.pop()
        self.a.append(a)
        return a

    def record(self, val: float):
        self.vals.append(val)

    def add_new_a(self):
        assert len(self.a_stack) == 0
        # sorts both lists according to the first one
        self.a, self.vals = map(list, zip(*sorted(zip(self.a, self.vals))))
        valnp = np.array(self.vals)
        fdeltas = np.abs(valnp[:-1] - valnp[1:])
        adeltas = np.array(self.a[1:]) - np.array(self.a[:-1])
        assert np.all(adeltas >= 0)
        step_too_high = (fdeltas > self.max_function_step).astype(int)
        new_a = step_too_high * adeltas / 2 + self.a[:-1]
        self.a_stack = new_a[step_too_high.astype(bool)].tolist()
