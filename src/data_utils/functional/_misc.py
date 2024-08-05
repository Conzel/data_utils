from typing import Iterator, TypeVar, Optional
from dataclasses import asdict, fields


def flatten_dict(config: dict) -> list[dict]:
    """Flattens a dict of lists into a list of dicts:
    Example:
        flatten_dict({'a': [1,2], 'b': [3,4]})
            = [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 4}, {'a': 2, 'b': 3}]
    """
    configs = [{}]
    for k, vs in config.items():
        if isinstance(vs, list):
            if len(vs) == 1:
                for conf in configs:
                    conf[k] = vs[0]
            elif len(vs) == 0:
                for conf in configs:
                    conf[k] = None
            else:
                new_configs = []
                for c in configs:
                    new_configs.extend([c.copy() for _ in range(len(vs))])
                configs = new_configs
                for i in range(len(configs)):
                    configs[i][k] = vs[i % len(vs)]
        else:
            for conf in configs:
                conf[k] = vs
    return configs


class dotdict(dict):
    """dot.notation access to dictionary attributes

    From https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


T = TypeVar("T")


def take(it: Iterator[T], n: int) -> Iterator[T]:
    for i, x in enumerate(it):
        if i < n:
            yield x
        else:
            break


def dataclass_list_bind(dataclasses: list) -> dotdict:
    # python black magic
    super_d = {}
    for d in dataclasses:
        for method in dir(d):
            if method.startswith("__") or callable(getattr(d, method)):
                continue

            super_d.setdefault(method, []).append(getattr(d, method))
    return dotdict(super_d)


def dataclass_from_dict(classname, d):
    field_set = {f.name for f in fields(classname) if f.init}
    filtered_args = {k: v for k, v in d.items() if k in field_set}
    return classname(**filtered_args)
