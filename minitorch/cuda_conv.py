from numba import cuda
import numpy as np
from minitorch.tensor import Tensor


def conv1d_cuda(input: Tensor, weight: Tensor) -> Tensor:
    pass
    # """Perform 1D convolution using CUDA."""
    # batch, in_channels, width = input.shape
    # out_channels, _, kernel_width = weight.shape
    # output_width = width - kernel_width + 1

    # # Allocate output tensor
    # output = Tensor.zeros((batch, out_channels, output_width), device="cuda")

    # # Define kernel
    # @cuda.jit
    # def conv1d_kernel(input, weight, output):
    #     b, oc, ow = cuda.grid(3)
    #     if b < batch and oc < out_channels and ow < output_width:
    #         result = 0.0
    #         for ic in range(in_channels):
    #             for kw in range(kernel_width):
    #                 result += input[b, ic, ow + kw] * weight[oc, ic, kw]
    #         output[b, oc, ow] = result

    # # Launch kernel
    # threads_per_block = (16, 16, 1)
    # blocks_per_grid = (
    #     (batch + threads_per_block[0] - 1) // threads_per_block[0],
    #     (out_channels + threads_per_block[1] - 1) // threads_per_block[1],
    #     (output_width + threads_per_block[2] - 1) // threads_per_block[2],
    # )
    # conv1d_kernel[blocks_per_grid, threads_per_block](
    #     input._storage, weight._storage, output._storage
    # )

    # return output


def conv2d_cuda(input: Tensor, weight: Tensor) -> Tensor:
    pass
    # """Perform 2D convolution using CUDA."""
    # batch, in_channels, height, width = input.shape
    # out_channels, _, kernel_height, kernel_width = weight.shape
    # output_height = height - kernel_height + 1
    # output_width = width - kernel_width + 1

    # # Allocate output tensor
    # output = Tensor.zeros((batch, out_channels, output_height, output_width), device="cuda")

    # # Define kernel
    # @cuda.jit
    # def conv2d_kernel(input, weight, output):
    #     b, oc, oh, ow = cuda.grid(4)
    #     if b < batch and oc < out_channels and oh < output_height and ow < output_width:
    #         result = 0.0
    #         for ic in range(in_channels):
    #             for kh in range(kernel_height):
    #                 for kw in range(kernel_width):
    #                     result += input[b, ic, oh + kh, ow + kw] * weight[oc, ic, kh, kw]
    #         output[b, oc, oh, ow] = result

    # # Launch kernel
    # threads_per_block = (8, 8, 8, 1)
    # blocks_per_grid = (
    #     (batch + threads_per_block[0] - 1) // threads_per_block[0],
    #     (out_channels + threads_per_block[1] - 1) // threads_per_block[1],
    #     (output_height + threads_per_block[2] - 1) // threads_per_block[2],
    #     (output_width + threads_per_block[3] - 1) // threads_per_block[3],
    # )
    # conv2d_kernel[blocks_per_grid, threads_per_block](
    #     input._storage, weight._storage, output._storage
    # )

    # return output
