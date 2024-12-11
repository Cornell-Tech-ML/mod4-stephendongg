# type: ignore
from typing import Tuple, TypeVar, Any


import numba
from numba import cuda
from numba.cuda import jit as _jit

import minitorch
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


# THREADS_PER_BLOCK = 16
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
    """CUDA 1D Convolution kernel."""
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    # Ensure the shapes match
    if not (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    ):
        return  # Alternatively, use assert for debugging

    # Define shared memory based on kernel width and input channels
    cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Each block handles one output position
    out_pos = cuda.blockIdx.x

    # Thread indices within the block
    pos_x = cuda.threadIdx.x  # Kernel width index
    pos_y = cuda.threadIdx.y  # Input channel index

    # Early exit if out_pos is out of bounds
    if out_pos >= out_size:
        return

    # Calculate the multi-dimensional index for the output
    batch_idx = out_pos // (out_channels * out_width)
    remainder = out_pos % (out_channels * out_width)
    out_channel_idx = remainder // out_width
    out_width_idx = remainder % out_width

    # Determine weight index and input position based on the reverse flag
    if reverse:
        weight_width_idx = kw - pos_x - 1
        input_pos = out_width_idx - weight_width_idx
    else:
        weight_width_idx = pos_x
        input_pos = out_width_idx + weight_width_idx

    # Load data into shared memory
    if pos_y < in_channels and pos_x < kw:
        if 0 <= input_pos < width:
            input_storage_idx = (
                batch_idx * input_strides[0]
                + pos_y * input_strides[1]
                + input_pos * input_strides[2]
            )
            inp = input[input_storage_idx]
        else:
            inp = 0.0

        weight_storage_idx = (
            out_channel_idx * weight_strides[0]
            + pos_y * weight_strides[1]
            + pos_x * weight_strides[2]
        )
        w = weight[weight_storage_idx] if pos_x < kw else 0.0

        cache[pos_x, pos_y] = inp * w
    else:
        cache[pos_x, pos_y] = 0.0

    # Synchronize to ensure all threads have written to shared memory
    cuda.syncthreads()

    # Accumulate results using a single thread
    if pos_x == 0 and pos_y == 0:
        accum = 0.0
        for i in range(kw):
            for j in range(in_channels):
                accum += cache[i, j]
        out[out_pos] = accum


tensor_conv1d = jit(_tensor_conv1d)


class Conv1dCudaFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute the forward pass of a 1D convolution.

        Args:
        ----
            ctx (Context): The context to save values for backpropagation.
            input (Tensor): Input tensor of shape (batch, in_channels, width).
            weight (Tensor): Weight tensor of shape (out_channels, in_channels, kernel_width).

        Returns:
        -------
            Tensor: The output tensor of shape (batch, out_channels, width).

        """
        # Save input and weight for the backward pass
        ctx.save_for_backward(input, weight)

        # Extract tensor shapes

        # batch_, out_channels, _, _ = output.shape

        batch, in_channels, width = input.shape
        out_channels, in_channels2, kernel_width = weight.shape

        assert in_channels == in_channels2

        # Output tensor shape (assume no padding or stride)
        # output = input.zeros((batch, out_channels, width))
        # output = output.contiguous()

        total_output_size = batch * out_channels * width

        output = TensorData(
            [0.0 for _ in range(total_output_size)], (batch, out_channels, width)
        )  # since its 1 dimensional!

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
        print("TEST HERE!!!")
        print(output.tuple())
        tensor_conv1d[blockspergrid, threadsperblock](
            output.tuple()[0],
            output.tuple()[1],
            output.tuple()[2],
            total_output_size,
            input._tensor._storage,
            input.shape,
            input._tensor._strides,
            weight._tensor._storage,
            weight.shape,
            weight._tensor._strides,
            False,
        )

        ret = Tensor.make(
            output.tuple()[0],
            tuple(output.tuple()[1]),
            tuple(output.tuple()[2]),
            backend=cuda_backend,
        )
        return ret

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass of 1D convolution.

        Args:
        ----
            ctx (Context): The context containing saved information for backpropagation.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the input tensor and weight tensor.

        """
        # Retrieve saved input and weight tensors
        input, weight = ctx.saved_values

        # Extract tensor shapes
        batch, in_channels, width = input.shape

        out_channels, in_channels2, kw = weight.shape

        assert in_channels == in_channels2

        # Compute gradients for the weight
        grad_weight = weight.zeros((in_channels, out_channels, kw))

        new_input = input.permute(
            1, 0, 2
        )  # Rearrange dimensions for compatibility: in channels, width, batch
        new_grad_output = grad_output.permute(1, 0, 2)  # Rearrange dimensions

        threadsperblock = (BLOCK_DIM, BLOCK_DIM)  # Match shared memory block size
        blockspergrid = grad_weight.size

        # Convolve input and grad_output to compute grad_weight
        tensor_conv1d[blockspergrid, threadsperblock](
            grad_weight.tuple()[0],
            grad_weight.tuple()[1],
            grad_weight.tuple()[2],
            grad_weight.size,
            new_input._tensor._storage,
            new_input._tensor.shape,
            new_input._tensor.strides,
            new_grad_output._tensor._storage,
            new_grad_output._tensor._shape,
            new_grad_output._tensor.strides,
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)  # Rearrange dimensions back

        # Compute gradients for the input
        grad_input = input.zeros((batch, in_channels, width))
        new_weight = weight.permute(1, 0, 2)  # Rearrange dimensions for compatibility

        threadsperblock = (BLOCK_DIM, BLOCK_DIM)  # Match shared memory block size
        blockspergrid = grad_input.size

        # Convolve grad_output and weight to compute grad_input
        tensor_conv1d[blockspergrid, threadsperblock](
            grad_input.tuple()[0],
            grad_input.tuple()[1],
            grad_input.tuple()[2],
            grad_input.size,
            grad_output._tensor._storage,
            grad_output._tensor.shape,
            grad_output._tensor.strides,
            new_weight._tensor._storage,
            new_weight._tensor.shape,
            new_weight._tensor.strides,
            True,  # Reverse flag for gradient computation
        )

        return grad_input, grad_weight


# Apply the function
conv1d = Conv1dCudaFun.apply


@cuda.jit
def _tensor_conv2d(
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
    pad_height: int,
    pad_width: int,
    reverse: bool,
) -> None:
    """CUDA 2D Convolution kernel with 'same' padding (pad only right and bottom)."""
    # Extract dimensions
    batch, out_channels, out_height, out_width = out_shape
    _, in_channels, height, width = input_shape
    _, _, kh, kw = weight_shape

    # Each block handles one output position
    out_pos = cuda.blockIdx.x

    # Thread indices within the block
    pos_y = cuda.threadIdx.y  # Kernel height index
    pos_x = cuda.threadIdx.x  # Kernel width index
    pos_c = cuda.threadIdx.z  # Input channel index

    # Early exit if out_pos is out of bounds
    if out_pos >= out_size:
        return

    # Calculate the multi-dimensional index for the output
    batch_idx = out_pos // (out_channels * out_height * out_width)
    remainder = out_pos % (out_channels * out_height * out_width)
    out_channel_idx = remainder // (out_height * out_width)
    remainder = remainder % (out_height * out_width)
    out_height_idx = remainder // out_width
    out_width_idx = remainder % out_width

    # # Calculate input indices with padding (pad_top=0, pad_left=0)
    # in_height_idx = out_height_idx + pos_y
    # in_width_idx = out_width_idx + pos_x

    # Calculate input indices with padding (pad_top=0, pad_left=0)
    if not reverse:
        in_height_idx = out_height_idx + pos_y
        in_width_idx = out_width_idx + pos_x
    else:
        # For grad_input, the relationship is different
        in_height_idx = out_height_idx - pos_y + (kh - 1)
        in_width_idx = out_width_idx - pos_x + (kw - 1)

    # Initialize shared memory as 1D array
    shared_cache = cuda.shared.array(
        shape=(7 * 7 * 64,), dtype=numba.float64
    )  # Adjust sizes as needed

    # Compute 1D index for shared_cache
    shared_cache_idx = pos_y * kw * in_channels + pos_x * in_channels + pos_c

    # Load data into shared memory
    if pos_c < in_channels and pos_y < kh and pos_x < kw:
        if 0 <= in_height_idx < height and 0 <= in_width_idx < width:
            input_storage_idx = (
                batch_idx * input_strides[0]
                + pos_c * input_strides[1]
                + in_height_idx * input_strides[2]
                + in_width_idx * input_strides[3]
            )
            inp = input[input_storage_idx]
        else:
            inp = 0.0  # Zero-padding

        weight_storage_idx = (
            out_channel_idx * weight_strides[0]
            + pos_c * weight_strides[1]
            + pos_y * weight_strides[2]
            + pos_x * weight_strides[3]
        )
        w = weight[weight_storage_idx]

        shared_cache[shared_cache_idx] = inp * w
    else:
        if pos_y < kh and pos_x < kw and pos_c < in_channels:
            shared_cache[shared_cache_idx] = 0.0

    # Synchronize to ensure all threads have written to shared memory
    cuda.syncthreads()

    # Perform the convolution sum
    # Only one thread (e.g., pos_x == 0 and pos_y == 0 and pos_c == 0) performs the accumulation
    if pos_x == 0 and pos_y == 0 and pos_c == 0:
        conv_sum = 0.0
        for c in range(in_channels):
            for i in range(kh):
                for j in range(kw):
                    idx = i * kw * in_channels + j * in_channels + c
                    conv_sum += shared_cache[idx]
        out[out_pos] = conv_sum


tensor_conv2d = jit(_tensor_conv2d)


# Define maximum sizes for shared memory
MAX_KERNEL_HEIGHT = 7  # Adjust as needed
MAX_KERNEL_WIDTH = 7  # Adjust as needed
MAX_IN_CHANNELS = 64  # Adjust as needed


class Conv2dCudaFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute the forward pass of a 2D convolution with 'same' padding."""
        # Save input and weight for the backward pass
        ctx.save_for_backward(input, weight)

        # Extract tensor shapes
        batch, in_channels, height, width = input.shape
        out_channels, in_channels2, kh, kw = weight.shape

        assert (
            in_channels == in_channels2
        ), "Input and weight channel dimensions must match."

        # Calculate padding for 'same' convolution
        pad_height = kh // 2
        pad_width = kw // 2

        # Calculate output dimensions for 'same' convolution
        out_height = height
        out_width = width

        # Total output size
        total_output_size = batch * out_channels * out_height * out_width

        # Initialize output tensor storage
        output_storage = [0.0 for _ in range(total_output_size)]
        output_shape = (batch, out_channels, out_height, out_width)

        output = TensorData(output_storage, output_shape)

        # Move output tensor to CUDA
        output.to_cuda_()

        # Define threads per block and blocks per grid
        # Each block handles one output position
        threadsperblock = (
            kh,
            kw,
            in_channels,
        )  # (kernel_height, kernel_width, in_channels)

        # Ensure that threadsperblock does not exceed CUDA's limit (1024)
        assert (
            kh * kw * in_channels <= 1024
        ), "Threads per block exceed CUDA's maximum limit."

        blockspergrid = (total_output_size,)

        # Launch the CUDA kernel with padding
        tensor_conv2d[blockspergrid, threadsperblock](
            output._storage,  # out: Storage (list of float)
            output.shape,  # out_shape: Shape (tuple of int)
            output.strides,  # out_strides: Strides (tuple of int)
            total_output_size,  # out_size: int
            input._tensor._storage,  # input: Storage (list of float)
            input._tensor.shape,  # input_shape: Shape (tuple of int)
            input._tensor.strides,  # input_strides: Strides (tuple of int)
            weight._tensor._storage,  # weight: Storage (list of float)
            weight._tensor.shape,  # weight_shape: Shape (tuple of int)
            weight._tensor.strides,  # weight_strides: Strides (tuple of int)
            pad_height,  # pad_height: int
            pad_width,  # pad_width: int
            False,  # reverse: bool (forward pass)
        )

        ret = Tensor.make(
            output.tuple()[0],
            tuple(output.tuple()[1]),
            tuple(output.tuple()[2]),
            backend=cuda_backend,
        )

        # # Wrap the output tensor
        # ret = Tensor.make(
        #     output._storage,             # Storage
        #     tuple(output.shape),         # Shape
        #     tuple(output.strides),       # Strides
        #     backend=cuda_backend,
        # )

        return ret

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass of 2D convolution."""
        # Retrieve saved input and weight tensors
        input, weight = ctx.saved_values

        # Extract tensor shapes
        batch, in_channels, height, width = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        _, _, out_height, out_width = grad_output.shape

        assert (
            in_channels == in_channels2
        ), "Input and weight channel dimensions must match."

        # Calculate padding for 'same' convolution
        # Calculate padding for 'same' convolution (pad only right and bottom)
        pad_bottom = kh - 1  # Pad only the bottom
        pad_right = kw - 1  # Pad only the right

        # For backward pass, set pad_height and pad_width to pad_top and pad_left
        pad_height = pad_bottom  # 0
        pad_width = pad_right  # 0

        # Compute gradients for the weight
        grad_weight_size = out_channels * in_channels * kh * kw
        grad_weight_storage = [0.0 for _ in range(grad_weight_size)]
        grad_weight_shape = (out_channels, in_channels, kh, kw)
        grad_weight = TensorData(grad_weight_storage, grad_weight_shape)
        grad_weight.to_cuda_()

        # Compute gradients for the input
        grad_input_size = batch * in_channels * height * width
        grad_input_storage = [0.0 for _ in range(grad_input_size)]

        grad_input_shape = (batch, in_channels, height, width)
        grad_input = TensorData(grad_input_storage, grad_input_shape)
        grad_input.to_cuda_()

        # Define threads per block and blocks per grid for grad_weight
        threadsperblock_w = (
            kh,
            kw,
            in_channels,
        )  # (kernel_height, kernel_width, in_channels)

        # Ensure that threadsperblock does not exceed CUDA's limit (1024)
        assert (
            kh * kw * in_channels <= 1024
        ), "Threads per block for grad_weight exceed CUDA's maximum limit."

        blockspergrid_w = (out_channels * batch * out_height * out_width,)

        # Launch the CUDA kernel for grad_weight
        tensor_conv2d[blockspergrid_w, threadsperblock_w](
            grad_weight._storage,  # out: Storage (list of float)
            grad_weight.shape,  # out_shape: Shape (tuple of int)
            grad_weight.strides,  # out_strides: Strides (tuple of int)
            grad_weight_size,  # out_size: int
            input._tensor._storage,  # input: Storage (list of float)
            input._tensor.shape,  # input_shape: Shape (tuple of int)
            input._tensor.strides,  # input_strides: Strides (tuple of int)
            grad_output._tensor._storage,  # weight: Storage (list of float)
            grad_output._tensor.shape,  # weight_shape: Shape (tuple of int)
            grad_output._tensor.strides,  # weight_strides: Strides (tuple of int)
            pad_height,  # pad_height: int
            pad_width,  # pad_width: int
            False,  # reverse: bool (grad_weight computation)
        )

        # Define threads per block and blocks per grid for grad_input
        threadsperblock_i = (
            kh,
            kw,
            out_channels,
        )  # (kernel_height, kernel_width, out_channels)

        # Ensure that threadsperblock does not exceed CUDA's limit (1024)
        assert (
            kh * kw * out_channels <= 1024
        ), "Threads per block for grad_input exceed CUDA's maximum limit."

        blockspergrid_i = (batch * in_channels * height * width,)

        # Launch the CUDA kernel for grad_input
        tensor_conv2d[blockspergrid_i, threadsperblock_i](
            grad_input._storage,  # out: Storage (list of float)
            grad_input.shape,  # out_shape: Shape (tuple of int)
            grad_input.strides,  # out_strides: Strides (tuple of int)
            grad_input_size,  # out_size: int
            weight._tensor._storage,  # input: Storage (list of float)
            weight._tensor.shape,  # input_shape: Shape (tuple of int)
            weight._tensor.strides,  # input_strides: Strides (tuple of int)
            grad_output._tensor._storage,  # weight: Storage (list of float)
            grad_output._tensor.shape,  # weight_shape: Shape (tuple of int)
            grad_output._tensor.strides,  # weight_strides: Strides (tuple of int)
            pad_height,  # pad_height: int
            pad_width,  # pad_width: int
            True,  # reverse: bool (grad_input computation)
        )

        a = Tensor.make(
            grad_input._storage,  # Storage
            tuple(grad_input.shape),  # Shape
            tuple(grad_input.strides),  # Strides
            backend=cuda_backend,
        )

        b = Tensor.make(
            grad_weight._storage,  # Storage
            tuple(grad_weight.shape),  # Shape
            tuple(grad_weight.strides),  # Strides
            backend=cuda_backend,
        )

        return a, b
