import datetime
from typing import Optional


class Timer:
    # Dictionary to store start times of multiple named timers
    _timers = {}

    @classmethod
    def start(cls, name):
        """Start a timer with the given name."""
        if name in cls._timers:
            raise ValueError(f"Timer '{name}' is already running.")
        cls._timers[name] = Timer(name)
        cls._timers[name].__enter__()

    @classmethod
    def end(cls, name):
        """End the timer with the given name and print the elapsed time."""
        if name not in cls._timers:
            raise ValueError(f"Timer '{name}' has not been started.")
        cls._timers[name].__exit__()
        del cls._timers[name]

    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __enter__(self):
        if self.name:
            print(f"Timer '{self.name}' started.")
        else:
            print(f"Timer started.")
        self.ts = datetime.datetime.now()

    def __exit__(self, *args):
        delta = datetime.datetime.now() - self.ts
        if self.name:
            print(f"{self.name}: Took {delta.total_seconds():.2f} seconds.")
        else:
            print(f"Took {delta.total_seconds():.2f} seconds.")
