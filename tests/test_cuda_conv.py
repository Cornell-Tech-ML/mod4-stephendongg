
import pytest
from hypothesis import given
from minitorch.tensor import Tensor
from cuda_conv import conv1d_cuda, conv2d_cuda
import torch
import numpy as np

@pytest.mark.task4_4b
def test_conv1d_cuda_simple():
    input_data = torch.randn(1, 1, 5, device="cuda")
    weight_data = torch.randn(1, 1, 3, device="cuda")
    input = Tensor(input_data.cpu().numpy(), device="cuda")
    weight = Tensor(weight_data.cpu().numpy(), device="cuda")

    # CUDA convolution
    output_cuda = conv1d_cuda(input, weight)

    # Torch convolution
    output_torch = torch.nn.functional.conv1d(input_data, weight_data)

    assert np.allclose(output_cuda.cpu().data, output_torch.cpu().numpy(), atol=1e-5)


@pytest.mark.task4_4b
def test_conv2d_cuda_simple():
    input_data = torch.randn(1, 1, 5, 5, device="cuda")
    weight_data = torch.randn(1, 1, 3, 3, device="cuda")
    input = Tensor(input_data.cpu().numpy(), device="cuda")
    weight = Tensor(weight_data.cpu().numpy(), device="cuda")

    # CUDA convolution
    output_cuda = conv2d_cuda(input, weight)

    # Torch convolution
    output_torch = torch.nn.functional.conv2d(input_data, weight_data)

    assert np.allclose(output_cuda.cpu().data, output_torch.cpu().numpy(), atol=1e-5)
