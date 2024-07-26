import datetime


class Timer:
    def __init__(self):
        pass

    def __enter__(self):
        self.ts = datetime.datetime.now()

    def __exit__(self, *args):
        delta = datetime.datetime.now() - self.ts
        print(f"Took {delta.total_seconds():.2f} seconds.")
