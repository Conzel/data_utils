import copy
from typing import Callable, Optional
import torch.nn as nn
import torch
from typing import overload


def _recursively_find_named_children(top_name: str, net: nn.Module):
    all_children = []
    if len(list(net.children())) == 0:
        return [(top_name, net)]
    for name, child in net.named_children():
        children_rec = _recursively_find_named_children(name, child)
        if top_name != "":
            children_rec = [
                (top_name + "." + child_name, child)
                for child_name, child in children_rec
            ]
        all_children.extend(children_rec)
    return all_children


def recursively_find_named_children(net: nn.Module):
    """Recursively find all named children of a network and returns an iterator (string, nn.Module)."""
    return _recursively_find_named_children("", net)


def map_net(net: nn.Module, f) -> nn.Module:
    """Apply a function to all layers of a network, statically. f accepts a module and modifies it in-place.

    The whole network is not modified in place but returned."""
    net = copy.deepcopy(net)
    for name, child in recursively_find_named_children(net):
        f(child)
    return net


def check_net_mappable(net: nn.Module, dummy_input: dict | torch.Tensor):
    """Check if a network can be using a hook function
    This is done by running the network on a dummy input and checking if we can detect any cyclical dependencies.
    """
    used_modules = [0]

    def detect(*args):
        used_modules[0] += 1

    children = recursively_find_named_children(net)
    handles = []
    for _, child in children:
        handles.append(child.register_forward_hook(detect))
    net(dummy_input)
    for handle in handles:
        handle.remove()
    return len(children) == used_modules[0]


def map_net_forward(
    net: nn.Module,
    x: dict | torch.Tensor,
    f,
    require_name: bool = False,
    sanity_check: bool = True,
    inplace: bool = False,
):
    """Apply a function to all layers of a network, while running an input x
    through the network. f has the same function signature as a hook.

    If require_name is set to true, the function f must have a name argument in the first place.
    """
    if inplace:
        net_mapped = net
    else:
        net_mapped = copy.deepcopy(net)

    handles = []
    named_children = recursively_find_named_children(net_mapped)
    for name, child in named_children:
        if require_name:
            f_curry = lambda name: lambda m, i, o: f(
                name, m, i, o
            )  # force evaluation of name
            handles.append(child.register_forward_hook(f_curry(name)))
        else:
            handles.append(child.register_forward_hook(f))

    if isinstance(x, torch.Tensor):
        net_mapped(x)
    else:
        net_mapped(**x)
    for handle in handles:
        handle.remove()
    return net_mapped


class Hooked:
    def __init__(
        self,
        net: torch.nn.Module,
        hook: Callable,
        filter: Callable,
        use_names: bool = False,
    ) -> None:
        self.net = net
        self.hook = hook
        self.filter = filter
        self.handles = []
        self.use_names = use_names

    def __enter__(self):
        for n, l in recursively_find_named_children(self.net):
            passed = self.filter(n, l) if self.use_names else self.filter(l)
            if passed:
                handle = l.register_forward_hook(self.hook)
                self.handles.append(handle)

    def __exit__(self, *_, **__):
        for handle in self.handles:
            handle.remove()
