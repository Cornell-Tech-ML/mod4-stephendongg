
import pytest
from hypothesis import given
import minitorch 
from minitorch.tensor import Tensor
from minitorch import conv1d_cuda, conv2d_cuda
import torch
import numpy as np
from .strategies import assert_close



cuda_backend = minitorch.TensorBackend(minitorch.CudaOps)
@pytest.mark.task4_4b
def test_conv1d_cuda_simple():

    input_data = [1, 1, 6]
    weight_data = [1, 1, 4]

    input = minitorch.tensor(input_data, backend=cuda_backend)
    weight = minitorch.tensor(weight_data, backend=cuda_backend)
    
    output_cuda = conv1d_cuda(input, weight)

    # Torch convolution
    output = minitorch.Conv1dFun.apply(input, weight)

    for i in range(output_cuda.size):
        assert_close(output_cuda[i], output[i])


# @pytest.mark.task4_4b
# def test_conv2d_cuda_simple():
#     input_data = torch.randn(1, 1, 5, 5, device="cuda")
#     weight_data = torch.randn(1, 1, 3, 3, device="cuda")
#     input = Tensor(input_data.cpu().numpy(), device="cuda")
#     weight = Tensor(weight_data.cpu().numpy(), device="cuda")

#     # CUDA convolution
#     output_cuda = conv2d_cuda(input, weight)

#     # Torch convolution
#     output_torch = torch.nn.functional.conv2d(input_data, weight_data)

#     assert np.allclose(output_cuda.cpu().data, output_torch.cpu().numpy(), atol=1e-5)
