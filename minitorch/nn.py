from typing import Tuple

from .tensor import Tensor
from .tensor_functions import rand, Max
from typing import Optional


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw
    # input = input.contiguous()
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = reshaped.permute(
        0, 1, 2, 4, 3, 5
    ).contiguous()  # use permute to move kh, kw into the last dimension
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)
    return tiled, new_height, new_width
    # raise NotImplementedError("Need to implement for Task 4.3")


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    tiled, new_height, new_width = tile(input, kernel)
    kh, kw = kernel
    pooled = tiled.mean(dim=4)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


# TODO: Implement for Task 4.3.
# def max(self, dim: Optional[int] = None) -> Tensor:
#     """Compute the max value along the specified dimension."""

#     if dim is None:
#         return Max.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
#     else:
#         return Max.apply(self, self._ensure_tensor(dim))
# return input.mul_reduce(dim, lambda x, y: operators.max(x, y))


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    # return Max.apply(input, tensor(dim))
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along the specified dimension."""
    max_val = input.max(dim)  # Subtract max for numerical stability
    exp_vals = (input - max_val).exp()
    return exp_vals / exp_vals.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along the specified dimension."""
    max_val = input.max(dim)  # Subtract max for numerical stability
    log_sum_exp = ((input - max_val).exp()).sum(dim).log()
    return input - max_val - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling using the specified kernel."""
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.max(dim=4)  # Apply max over kernel dimension
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(
    input: Tensor, rate: float, train: bool = True, ignore: bool = False
) -> Tensor:
    if ignore:
        return input  # Simply return the input unchanged when ignore is True
    if not train:
        return input
    if (
        rate >= 1.0
    ):  # Dropout rate of 1.0 means everything is dropped (output all zeros)
        return input * 0  # Scale by zero tensor
    if rate <= 0.0:  # Dropout rate of 0.0 means nothing is dropped (return input)
        return input
    mask = rand(input.shape) > rate
    scaled = mask * (1.0 / (1.0 - rate))  # Scale by the inverse of keep probability
    return input * scaled
