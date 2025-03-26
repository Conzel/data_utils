from typing import Literal, Optional
import numpy as np


class AdaptiveSampler:
    """Adaptive sampler for probing a **monotonous** functional f: R -> R with respect to a parameter. The sampler can be used
    as an iterator which produces new parameter recommendations.

    Usage example:
    s = AdaptiveSampler((1, 100), max_function_step=0.1, output_range=(-4, 4))
    f_sampled = []
    params = []

    for p in s:
        output = f(p)
        f_sampled.append(output)
        params.append(p)
        s.record(output)

    If the sampler finishes before the maximum number of iterations (default: 100), it is guaranteed
    that the specified output range is covered with a granularity of 0.1 (maximum dist between values).

    input_range: Specify minimum and maximum value of the input
    output_range: Specify a region of interest. If this is None, f(input_range[0]) and f(input_range[1]) are used to determine the range
    initial_sampling: Specify how to determine the initial sampling points. Does a linear/logarithmic covering of the input range
    initial_sampling_points: How many points the initial sampling contains
    """

    def __init__(
        self,
        input_range: tuple[float, float],
        max_function_step: float,
        max_iter=100,
        output_range: Optional[tuple[float, float]] = None,
        initial_sampling: Literal["linear", "log"] = "log",
        initial_sampling_points=10,
    ):
        self.output_range = output_range
        self.input_range = input_range
        self.max_function_step = max_function_step
        self.max_iter = max_iter
        self.initial_sampling: Literal["linear", "log"] = initial_sampling
        self.initial_sampling_points = initial_sampling_points
        self.refresh()

    def refresh(
        self,
    ):

        a0, a1 = self.input_range
        if self.initial_sampling == "log" and a0 <= 0:
            raise ValueError("Logarithmic sampling requires positive range.")

        if self.initial_sampling == "log":
            self.a_stack = np.logspace(
                np.log10(a0), np.log10(a1), self.initial_sampling_points
            ).tolist()
        else:
            self.a_stack = np.linspace(a0, a1, self.initial_sampling_points).tolist()
        self.vals = []
        self.a = []
        self.max_function_step = self.max_function_step
        self.i = 0
        self.max_iter = self.max_iter

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
        anp = np.array(self.a)
        if self.output_range is not None:
            inside_range = np.logical_and(
                valnp >= self.output_range[0], valnp <= self.output_range[1]
            )
            first_over = np.argmax(inside_range)
            if first_over > 0:
                inside_range[first_over - 1] = True
            last_over = len(inside_range) - np.argmax(inside_range[::-1]) - 1
            if last_over < len(inside_range) - 1:
                inside_range[last_over + 1] = True
            valnp = valnp[inside_range]
            anp = anp[inside_range]
        fdeltas = np.abs(valnp[:-1] - valnp[1:])
        adeltas = anp[1:] - anp[:-1]
        assert np.all(adeltas >= 0)
        step_too_high = (fdeltas > self.max_function_step).astype(int)
        new_a = step_too_high * adeltas / 2 + anp[:-1]
        self.a_stack = new_a[step_too_high.astype(bool)].tolist()

    def __iter__(self):
        """Return self as iterator."""
        self.refresh()
        return self

    def __next__(self):
        """Get the next parameter value."""
        next_val = self.get_next()
        if next_val is None:
            raise StopIteration
        self._current_value = next_val
        return next_val
