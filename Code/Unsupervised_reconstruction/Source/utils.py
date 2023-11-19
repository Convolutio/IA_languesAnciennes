import torch
from torch.nn import Module
from torch import Tensor
from torch import device
from torch.types import Device
from typing import TypeVar, Union


def pad2d_sequence(sequence: list[Tensor], padding_value: float):
    """
    Let sequence a list of n tensors of shape (~x, ~y, *), with ~l a variable length, * a uniform shape.
    Return a padded tensor of shape (max_x, max_y, n, *)
    """
    maxX = max([t.size()[0] for t in sequence])
    maxY = max([t.size()[1] for t in sequence])
    padded = torch.full((maxX, maxY, 
                         len(sequence), *sequence[0].size()[2:]),
                         padding_value, dtype=sequence[0].dtype)

    for (i, t) in enumerate(sequence):
        x_l, y_l = t.size()[:2]
        padded[:x_l, :y_l, i] = t

    return padded


def computePaddingMask(sourceLengthData: tuple[Tensor, int], targetLengthData: tuple[Tensor, int], device = None):
    """
    Computes a mask which equals False at the positions of padding tokens and True anywhere else.

    The expected length data for the sequences x (samples) and y (cognates) will determine the shape of the returned tensor.  
    Mask dim = (L_x, L_y, *), with * some shape that of the two tensors in the arguments.

    >>> sourceLengths = (torch.tensor([5, 3, 2, 3]), 5)
    >>> targetLengths = (torch.tensor([4, 8, 5, 6]), 8)
    >>> mask = self.__computeMask(sourcesLengths, targetLengths)
    >>> mask.size()
    torch.Size([5, 8, 4])
    >>> mask[..., 0]
    tensor([[ True,  True,  True,  True, False, False, False, False],
            [ True,  True,  True,  True, False, False, False, False],
            [ True,  True,  True,  True, False, False, False, False],
            [ True,  True,  True,  True, False, False, False, False],
            [ True,  True,  True,  True, False, False, False, False]])
    >>> mask[..., 1]
    tensor([[ True,  True,  True,  True,  True,  True,  True,  True],
            [ True,  True,  True,  True,  True,  True,  True,  True],
            [ True,  True,  True,  True,  True,  True,  True,  True],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]])
    >>> mask[..., 2]
    tensor([[ True,  True,  True,  True,  True, False, False, False],
            [ True,  True,  True,  True,  True, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]])
    >>> masked_ctx = mask*ctx

    Args:
        * sourceLengthData: a tuple with the following elements:
            - sourcesLengths: a cpu tensor containing the wanted lengths of the samples (`L_x`) (shape = *)
            - maxSourceLength: the value of the maximum length.
        * targetLengthData: if None, this data is not computed and got from the cache. Else, the tuple with the following elements must be given:
            - targetsLengths: a cpu tensor containing the wanted lengths of the modern forms (`L_y`) (shape = *)
            - maxTargetLength: the value of the maximum length
        * device: device on which the returned tensor will be placed.
    """
    sourceLengths, maxSourceLength = sourceLengthData
    targetLengths, maxTargetLength = targetLengthData

    unsqueezing_tuple = (...,) + (None,)*sourceLengths.ndim

    A = torch.arange(maxSourceLength)[
        unsqueezing_tuple] < sourceLengths.unsqueeze(0)  # dim = (L_x, *)
    B = torch.arange(maxTargetLength)[
        unsqueezing_tuple] < targetLengths.unsqueeze(0)  # dim = (L_y, *)

    # dim = (L_x, L_y, *)
    return torch.logical_and(A.unsqueeze(1), B.unsqueeze(0)).to(device)


def _getDevice(obj) -> Union[device, None]:
    if isinstance(obj, Tensor):
        return obj.device

    if isinstance(obj, device):
        return obj

    if isinstance(obj, Module) or issubclass(type(obj), Module):
        # Return the device name for the first parameter of the nn module
        return next(obj.parameters()).device

    return None


#TODO: Test this!
def checkSameDevice(x, y, *args) -> Union[None, Device]:
    """
    If the device of `x` is the same as `y` (and `args` if they exist),
    return the device name, otherwise return `None`.
    """
    if not (device := _getDevice(x)):
        return None

    if not (device == _getDevice(y)):
        return None

    for arg in args:
        if not (device == _getDevice(arg)):
            return None

    return device


__KeyType = TypeVar('__KeyType')
__ValType = TypeVar('__ValType')


def dl_to_ld(dl: dict[__KeyType, list[__ValType]]) -> list[dict[__KeyType, __ValType]]:
    new_list = [{key: dl[key][i] for key in dl}
                for i in range(len(dl[next(iter(dl))]))]
    return new_list


__KeyType2 = TypeVar('__KeyType2')
__ValType2 = TypeVar('__ValType2')


def ld_to_dl(ld: list[dict[__KeyType2, __ValType2]]) -> dict[__KeyType2, list[__ValType2]]:
    list_length = len(ld)
    new_dict = {key: [ld[i][key] for i in range(list_length)] for key in ld[0]}
    return new_dict