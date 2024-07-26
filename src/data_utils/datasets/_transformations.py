from torch.utils.data import Dataset
import torch


def shuffled(dataset: Dataset):
    """Returns a shuffle version of the dataset.

    WARNING: Do not re-shuffle a dataset multiple times, as the shuffling is implemented
    by access to a random permutation matrix. If you do this multiple times, each data-access requires multiple
    accesses to the matrix."""

    class _Shuffled(Dataset):
        def __init__(self, dataset):
            # choose a temporary name no one would use, as this
            # should not clash with dataset attributes
            self._sJSDOFUhwqnsdofuh2973sndfovuOUSJH = dataset
            self.perm = torch.randperm(dataset.__len__())

        def __getitem__(self, index):
            return self._sJSDOFUhwqnsdofuh2973sndfovuOUSJH.__getitem__(self.perm[index])

        def __len__(self):
            # magic methods cannot be explicitly propagated
            return len(self._sJSDOFUhwqnsdofuh2973sndfovuOUSJH)

        def __getattr__(self, name: str):
            # if name == "__getitem__":
            #    return self.__getitem__
            # else:
            return getattr(self._sJSDOFUhwqnsdofuh2973sndfovuOUSJH, name)

    return _Shuffled(dataset)
