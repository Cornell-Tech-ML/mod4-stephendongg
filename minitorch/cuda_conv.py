# type: ignore
from asyncio import threads
from typing import Tuple, TypeVar, Any

import math

import numba 
from numba import njit as _njit, prange
from numba import cuda 
from numba.cuda import jit as _jit

import minitorch
from minitorch.cuda_ops import THREADS_PER_BLOCK
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
    TensorData,
    MAX_DIMS
)
from .tensor_functions import Function

Fn = TypeVar("Fn")

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")
cuda_backend = minitorch.TensorBackend(minitorch.CudaOps)

# Block size for shared memory
BLOCK_DIM = 32



def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT-compile a function for use on CUDA devices.

    Args:
    ----
        fn (Fn): The function to compile for CUDA.
        **kwargs: Additional arguments passed to the Numba `jit` function.

    Returns:
    -------
        Fn: The compiled CUDA device function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT-compile a CUDA kernel function.

    Args:
    ----
        fn: The function to compile as a CUDA kernel.
        **kwargs: Additional arguments passed to the Numba `jit` function.

    Returns:
    -------
        FakeCUDAKernel: The compiled CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 16

@cuda.jit
def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    CUDA 1D Convolution kernel (refactored).

    Args:
        out: Output tensor storage.
        out_shape: Shape of the output tensor.
        out_strides: Strides for the output tensor.
        out_size: Total size of the output tensor.
        input: Input tensor storage.
        input_shape: Shape of the input tensor.
        input_strides: Strides for the input tensor.
        weight: Storage for the weight tensor.
        weight_shape: Shape of the weight tensor.
        weight_strides: Strides for the weight tensor.
        reverse: Whether to reverse the kernel.
    """
    # Extract dimensions
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, k_width = weight_shape

    # Ensure shape consistency
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Allocate shared memory
    cache = cuda.shared.array((32, 32), dtype=numba.float32)
    out_index = cuda.local.array(MAX_DIMS, dtype=numba.int32)
    out_pos = cuda.blockIdx.x
    pos_x = cuda.threadIdx.x
    pos_y = cuda.threadIdx.y

    # Process each output position
    if out_pos < out_size:
        to_index(out_pos, out_shape, out_index)
        batch_index = out_index[0]
        out_channel_index = out_index[1]
        out_width_index = out_index[2]

        in_channel_index = pos_x
        kernel_offset = pos_y

        if in_channel_index < in_channels and kernel_offset < k_width:
            input_val = 0.0
            weight_val = 0.0
            if reverse:
                kernel_index = -kernel_offset
            else:
                kernel_index = kernel_offset

            # Check valid input positions
            if (
                out_width_index + kernel_index >= 0
                and out_width_index + kernel_index < width
            ):
                input_pos = (
                    batch_index * input_strides[0]
                    + in_channel_index * input_strides[1]
                    + (out_width_index + kernel_index) * input_strides[2]
                )
                weight_pos = (
                    out_channel_index * weight_strides[0]
                    + in_channel_index * weight_strides[1]
                    + kernel_offset * weight_strides[2]
                )
                input_val = input[input_pos]
                weight_val = weight[weight_pos]

            # Store intermediate results in shared memory
            cache[in_channel_index, kernel_index] = input_val * weight_val
            cuda.syncthreads()

            # Aggregate results
            if in_channel_index == 0 and kernel_index == 0:
                value = 0.0
                for c in range(in_channels):
                    for k in range(k_width):
                        value += cache[c, k]
                out[out_pos] = value


tensor_conv1d = cuda.jit()(_tensor_conv1d)


# @cuda.jit
# def _tensor_conv1d(
#     out: Storage, out_shape: Shape, out_strides: Strides, out_size: int,
#     input: Storage, input_shape: Shape, input_strides: Strides, 
#     weight: Storage, weight_shape: Shape, weight_strides: Strides,
#     reverse: bool,
# ) -> None:
#     """
#     CUDA 1D Convolution kernel.

#     Args:
#         out: Output tensor storage.
#         out_shape: Shape of the output tensor.
#         out_strides: Strides for the output tensor.
#         out_size: Total size of the output tensor.
#         input: Input tensor storage.
#         input_shape: Shape of the input tensor.
#         input_strides: Strides for the input tensor.
#         weight: Storage for the weight tensor.
#         weight_shape: Shape of the weight tensor.
#         weight_strides: Strides for the weight tensor.
#         reverse: Whether to reverse the kernel.
#     """

#     # Shared memory declaration
#     shared_input = cuda.shared.array((BLOCK_DIM,), numba.float32)
#     shared_weight = cuda.shared.array((BLOCK_DIM,), numba.float32)

#     # Thread and block indices
#     out_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     thread_idx = cuda.threadIdx.x

#     if out_idx < out_size:
#         # Load data into shared memory
#         if thread_idx < input_shape[2]:
#             shared_input[thread_idx] = input[thread_idx]
#         else:
#             shared_input[thread_idx] = 0.0

#         if thread_idx < weight_shape[2]:
#             shared_weight[thread_idx] = weight[thread_idx]
#         else:
#             shared_weight[thread_idx] = 0.0

#         # Synchronize threads to ensure data is loaded
#         cuda.syncthreads()

#         # Compute the convolution
#         result = 0.0
#         for k in range(weight_shape[2]):
#             if reverse:
#                 weight_idx = weight_shape[2] - 1 - k
#             else:
#                 weight_idx = k

#             if thread_idx + k < input_shape[2]:
#                 result += shared_input[thread_idx + k] * shared_weight[weight_idx]

#         # Write result to the output tensor
#         out[out_idx] = result

# tensor_conv1d = jit(_tensor_conv1d)
class Conv1dCudaFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute the forward pass of a 1D convolution.

        Args:
            ctx (Context): The context to save values for backpropagation.
            input (Tensor): Input tensor of shape (batch, in_channels, width).
            weight (Tensor): Weight tensor of shape (out_channels, in_channels, kernel_width).

        Returns:
            Tensor: The output tensor of shape (batch, out_channels, width).
        """
        # Save input and weight for the backward pass
        ctx.save_for_backward(input, weight)

        # Extract tensor shapes
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kernel_width = weight.shape
        assert in_channels == in_channels2

        # Output tensor shape (assume no padding or stride)
        # output = input.zeros((batch, out_channels, width))
        # output = output.contiguous()

        output = TensorData([0.0 for _ in range(width)], (batch, in_channels, width)) # since its 1 dimensional! 

        # print(output.strides)
        output.to_cuda_()


        # TODO: Define threads per block and blocks per grid. 
        # threadsperblock = 1 # Common choice, depends on GPU
        # # blockspergrid = (width + threadsperblock - 1) // threadsperblock
        # threadsperblock = (BLOCK_DIM, BLOCK_DIM)
        # blockspergrid_x = (output.shape[-2] + BLOCK_DIM - 1) // BLOCK_DIM
        # blockspergrid_y = (output.shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM

        # blockspergrid = (blockspergrid_x, blockspergrid_y, output.shape[0])

        threadsperblock = (BLOCK_DIM, BLOCK_DIM)  # Match shared memory block size
        blockspergrid = output.size

        # # Run the CUDA 1D convolution kernel
        # tensor_conv1d[blockspergrid, threadsperblock](
        #     output._tensor._storage, output.shape, output._tensor._strides, output.size,
        #     input._tensor._storage, input.shape, input._tensor._strides,
        #     weight._tensor._storage, weight.shape, weight._tensor._strides,
        #     False,  # Reverse flag
        # )
        # print()
        # print("1: Initial Checks")
        # print("Input Shape:", input.shape, "Strides:", input._tensor._strides)
        # print("Weight Shape:", weight.shape, "Strides:", weight._tensor._strides)
        # print("Output Shape:", output.shape, "Strides:", output.strides)

        # print()
        # print("2: Final Checks")
        # print("Output Tensor Before Kernel:")
        # print(output._storage)

        out = Tensor.make(output.tuple()[0], tuple(output.tuple()[1]), tuple(output.tuple()[2]), backend=cuda_backend)

        tensor_conv1d[blockspergrid, threadsperblock](
            out.tuple()[0], out.tuple()[1], out.tuple()[2], width,
            input._tensor._storage, input.shape, input._tensor._strides,
            weight._tensor._storage, weight.shape, weight._tensor._strides,
            False,
        )
        # After kernel execution
        # print("Output Tensor After Kernel:")
        # print(output._storage)
        
        # print()
        # print("3. Kernel launch parameters")
        # print("CUDA Kernel Launch Parameters:")
        # print("Threads per Block:", threadsperblock)
        # print("Blocks per Grid:", blockspergrid)

        # print("Input Tensor Storage:")
        # print(input._tensor._storage)
        # print("Weight Tensor Storage:")
        # print(weight._tensor._storage)

        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the backward pass of 1D convolution.

        Args:
            ctx (Context): The context containing saved information for backpropagation.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
            Tuple[Tensor, Tensor]: Gradients of the input tensor and weight tensor.
        """
        # Retrieve saved input and weight tensors
        input, weight = ctx.saved_values

        # Extract tensor shapes
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kernel_width = weight.shape
        assert in_channels == in_channels2

        # Compute gradients for the weight
        grad_weight = weight.zeros((in_channels, out_channels, kernel_width))

        new_input = input.permute(1, 0, 2)  # Rearrange dimensions for compatibility
        new_grad_output = grad_output.permute(1, 0, 2)  # Rearrange dimensions

        threadsperblock = (BLOCK_DIM, BLOCK_DIM)  # Match shared memory block size
        blockspergrid = grad_weight.size

        # Convolve input and grad_output to compute grad_weight
        tensor_conv1d[blockspergrid, threadsperblock](
            grad_weight.tuple()[0], grad_weight.tuple()[1], grad_weight.tuple()[2], grad_weight.size,
            new_input._tensor._storage, new_input._tensor.shape, new_input._tensor.strides,
            new_grad_output._tensor._storage, new_grad_output._tensor._shape, new_grad_output._tensor.strides,
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)  # Rearrange dimensions back

        # Compute gradients for the input
        grad_input = input.zeros(input.shape)
        new_weight = weight.permute(1, 0, 2)  # Rearrange dimensions for compatibility

        threadsperblock = (BLOCK_DIM, BLOCK_DIM)  # Match shared memory block size
        blockspergrid = grad_weight.size
        
        # Convolve grad_output and weight to compute grad_input
        tensor_conv1d[blockspergrid, threadsperblock](
            grad_input.tuple()[0], grad_input.tuple()[1], grad_input.tuple()[2], grad_input.size,
            grad_output._tensor._storage, grad_output._tensor.shape, grad_output._tensor.strides,
            new_weight._tensor._storage, new_weight._tensor.shape, new_weight._tensor.strides,
            True,  # Reverse flag for gradient computation
        )

        return grad_input, grad_weight


# Apply the function
conv1d = Conv1dCudaFun.apply




# def _tensor_conv2d(
#     out: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     input: Storage,
#     input_shape: Shape,
#     input_strides: Strides,
#     weight: Storage,
#     weight_shape: Shape,
#     weight_strides: Strides,
#     reverse: bool,
# ) -> None:
#     """2D Convolution implementation.

#     Given input tensor of

#        `batch, in_channels, height, width`

#     and weight tensor

#        `out_channels, in_channels, k_height, k_width`

#     Computes padded output of

#        `batch, out_channels, height, width`

#     `Reverse` decides if weight is anchored top-left (False) or bottom-right.
#     (See diagrams)


#     Args:
#     ----
#         out (Storage): storage for `out` tensor.
#         out_shape (Shape): shape for `out` tensor.
#         out_strides (Strides): strides for `out` tensor.
#         out_size (int): size of the `out` tensor.
#         input (Storage): storage for `input` tensor.
#         input_shape (Shape): shape for `input` tensor.
#         input_strides (Strides): strides for `input` tensor.
#         weight (Storage): storage for `input` tensor.
#         weight_shape (Shape): shape for `input` tensor.
#         weight_strides (Strides): strides for `input` tensor.
#         reverse (bool): anchor weight at top-left or bottom-right

#     """
#     batch_, out_channels, _, _ = out_shape
#     batch, in_channels, height, width = input_shape
#     out_channels_, in_channels_, kh, kw = weight_shape

#     assert (
#         batch == batch_
#         and in_channels == in_channels_
#         and out_channels == out_channels_
#     )

#     s1 = input_strides
#     s2 = weight_strides
#     # inners
#     s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
#     s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

#     for b in prange(batch_):
#         for oc in prange(out_channels):
#             for h_out in prange(out_shape[2]):  # iterate over output height
#                 for w_out in prange(out_shape[3]):  # iterate over output width
#                     out_value = 0.0

#                     for ic in prange(in_channels):  # iterate over input channels
#                         for kh_ in prange(kh):  # Iterate over kernal height
#                             for kw_ in prange(kw):  # Iterate over kernel weidth
#                                 h_in = h_out - kh_ if reverse else h_out + kh_
#                                 w_in = w_out - kw_ if reverse else w_out + kw_

#                                 if 0 <= h_in < height and 0 <= w_in < width:
#                                     input_idx = (
#                                         b * s10 + ic * s11 + h_in * s12 + w_in * s13
#                                     )
#                                     weight_idx = (
#                                         oc * s20 + ic * s21 + kh_ * s22 + kw_ * s23
#                                     )
#                                     out_value += input[input_idx] * weight[weight_idx]

#                     out_idx = (
#                         b * out_strides[0]
#                         + oc * out_strides[1]
#                         + h_out * out_strides[2]
#                         + w_out * out_strides[3]
#                     )
#                     out[out_idx] = out_value


# tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


# class Conv2dFun(Function):
#     @staticmethod
#     def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
#         """Compute a 2D Convolution

#         Args:
#         ----
#             ctx : Context
#             input : batch x in_channel x h x w
#             weight  : out_channel x in_channel x kh x kw

#         Returns:
#         -------
#             (:class:`Tensor`) : batch x out_channel x h x w

#         """
#         ctx.save_for_backward(input, weight)
#         batch, in_channels, h, w = input.shape
#         out_channels, in_channels2, kh, kw = weight.shape
#         assert in_channels == in_channels2
#         output = input.zeros((batch, out_channels, h, w))
#         tensor_conv2d(
#             *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
#         )
#         return output

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
#         """Compute the backward pass of 2D convolution.

#         Args:
#         ----
#             ctx (Context): The context containing saved information for backpropagation.
#             grad_output (Tensor): The gradient of the output tensor.

#         Returns:
#         -------
#             Tuple[Tensor, Tensor]: Gradients of the input tensor and weight tensor.

#         """
#         input, weight = ctx.saved_values
#         batch, in_channels, h, w = input.shape
#         out_channels, in_channels, kh, kw = weight.shape

#         grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
#         new_input = input.permute(1, 0, 2, 3)
#         new_grad_output = grad_output.permute(1, 0, 2, 3)
#         tensor_conv2d(  # type: ignore
#             *grad_weight.tuple(),
#             grad_weight.size,
#             *new_input.tuple(),
#             *new_grad_output.tuple(),
#             False,  # type: ignore
#         )
#         grad_weight = grad_weight.permute(1, 0, 2, 3)

#         grad_input = input.zeros((batch, in_channels, h, w))
#         new_weight = weight.permute(1, 0, 2, 3)
#         tensor_conv2d(  # type: ignore
#             *grad_input.tuple(),
#             grad_input.size,  # type: ignore
#             *grad_output.tuple(),
#             *new_weight.tuple(),
#             True,  # type: ignore
#         )
#         return grad_input, grad_weight


# conv2d = Conv2dFun.apply
