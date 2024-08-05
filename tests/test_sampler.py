from data_utils.experiments import AdaptiveSampler
import numpy as np


def test_adaptive_sampler():
    f = lambda x: x**2

    fvals = []
    sampler = AdaptiveSampler(
        (1, 3),
        0.05,
        max_iter=300,
        initial_sampling="linear",
        initial_sampling_points=20,
    )
    while x := sampler.get_next():
        fvals.append(f(x))
        sampler.record(fvals[-1])

    fvals = np.array(sorted(fvals))
    diffs = np.abs(fvals[:-1] - fvals[1:])
    assert np.all(diffs < 0.5), f"Max Diff: {diffs.max()}"

    assert np.allclose(np.array(sampler.a) ** 2, fvals)
