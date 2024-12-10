from typing import Tuple

from .tensor import Tensor
from .tensor_functions import rand
from typing import Optional
from .tensor_functions import Function, tensor

from .autodiff import Context


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor (Go back to argmax)
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

    new_height = height // kh
    new_width = width // kw
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = reshaped.permute(
        0, 1, 2, 4, 3, 5
    ).contiguous()  # use permute to move kh, kw into the last dimension
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)
    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling using the specified kernel.

    Args:
    ----
        input (Tensor): The input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): The height and width of the pooling kernel.

    Returns:
    -------
        Tensor: The pooled tensor of shape (batch, channel, new_height, new_width).

    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=4)
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


class Max(Function):
    """A function to compute the maximum value along a specified dimension."""

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass of sum.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            a (Tensor): Input tensor.
            dim (Tensor): The dimension to reduce.

        Returns:
        -------
            Tensor: The result of sum `a`.

        """
        ctx.save_for_backward(a, dim)  # use a mask in order to retrieve it
        return a.f.max_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass of sum.

        Args:
        ----
            ctx (Context): The context to store information for backpropagation.
            grad_output (Tensor): Gradient of the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradient of the input and the derivative of the loss with respect to the dim.

        """
        a, dim = ctx.saved_values
        one_hot = argmax(a, int(dim.item()))
        return grad_output * one_hot, tensor([0.0])


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (Optional[int]): The dimension to compute the max over. If None, computes the max over all elements.

    Returns:
    -------
        Tensor: The maximum values along the specified dimension.

    """
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def argmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (Optional[int]): The dimension to compute the argmax over. If None, computes the argmax over all elements.

    Returns:
    -------
        Tensor: A 1-hot tensor indicating the positions of the maximum values.

    """
    if dim is None:
        out = input.f.max_reduce(input, 0)
    else:
        out = input.f.max_reduce(input, int(input._ensure_tensor(dim).item()))
    return out == input


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along the specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to compute the softmax over.

    Returns:
    -------
        Tensor: The softmax values along the specified dimension.

    """
    max_val = max(input, dim)
    exp_vals = (input - max_val).exp()
    return exp_vals / exp_vals.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along the specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to compute the logsoftmax over.

    Returns:
    -------
        Tensor: The log of the softmax values along the specified dimension.

    """
    max_val = max(input, dim)  # Subtract max for numerical stability
    log_sum_exp = ((input - max_val).exp()).sum(dim).log()
    return input - max_val - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling using the specified kernel.

    Args:
    ----
        input (Tensor): The input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): The height and width of the pooling kernel.

    Returns:
    -------
        Tensor: The pooled tensor of shape (batch, channel, new_height, new_width).

    """
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max(tiled, dim=4)  # Apply max over kernel dimension
    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor based on the specified rate.

    Args:
    ----
        input (Tensor): The input tensor.
        rate (float): The dropout rate (probability of dropping elements).
        ignore (bool): If True, disables dropout and returns the input unchanged.

    Returns:
    -------
        Tensor: The tensor after applying dropout.

    """
    if ignore:
        return input  # Simply return the input unchanged when ignore is True
    if (
        rate >= 1.0
    ):  # Dropout rate of 1.0 means everything is dropped (output all zeros)
        return input * 0  # Scale by zero tensor
    if rate <= 0.0:  # Dropout rate of 0.0 means nothing is dropped (return input)
        return input
    mask = rand(input.shape) > rate
    return input * mask
