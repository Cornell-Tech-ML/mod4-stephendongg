
import pytest
from hypothesis import given, settings
from hypothesis.strategies import lists, floats
import minitorch 
from minitorch.tensor import Tensor
import torch
import numpy as np
from .strategies import assert_close

from typing import List

# Define small floats for testing
small_floats = floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)




cuda_backend = minitorch.TensorBackend(minitorch.CudaOps)
simple_backend = minitorch.TensorBackend(minitorch.SimpleOps)
@pytest.mark.task4_4b
def test_conv1d_cuda_simple():

    input_data = [1, 1, 1, 1]
    weight_data = [1, 1, 4]
    
    input = minitorch.tensor(input_data, backend=cuda_backend).view(1, 1, 4)
    weight = minitorch.tensor(weight_data, backend=cuda_backend).view(1, 1, 3)

    
    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 3), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4), backend=simple_backend)
    weight_simple = Tensor.make(np.array(weight_data), (1, 1, 3), backend=simple_backend)
    
    output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)

    # Torch convolution
    output = minitorch.Conv1dFun.apply(input_simple, weight_simple)


    for i in range(output_cuda.size):
        assert_close(output_cuda._tensor._storage[i], output._tensor._storage[i])



@pytest.mark.task4_4b
def test_conv1d_zero_weight_cuda_simple():
    """Test for 1D convolution with all-zero weights comparing CUDA and Simple backend storages."""
    input_data = [1, 2, 3, 4]
    weight_data = [0, 0, 0]

    # Create tensors
    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 3), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4), backend=simple_backend)
    weight_simple = Tensor.make(np.array(weight_data), (1, 1, 3), backend=simple_backend)

    # Compute outputs
    output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)
    output_simple = minitorch.Conv1dFun.apply(input_simple, weight_simple)

    # Compare storages
    for i in range(output_cuda.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])


@pytest.mark.parametrize(
    "input_data, weight_data, expected",
    [
        # Simple Cases
        ([1, 2, 3, 4], [0, 0, 0], [0, 0]),  # All-zero weights
        ([1, 2, 3, 4], [1, 0, 0], [1, 2]),  # Single active weight
        ([1, 1, 1, 1], [1, 1, 1], [3, 3]),  # Uniform weights
        ([1, -1, 1, -1], [1, -1, 1], [1, -1]),  # Alternating values

        # Edge Cases
        ([0, 0, 0, 0], [0, 0, 0], [0, 0]),  # All-zero inputs and weights
        ([1, 1, 1, 1], [1, -1, 1], [1, 1]),  # Weights sum to 1
        ([10, 20, 30, 40], [1, 2, 3], [140, 200]),  # Larger inputs
    ],
)
def test_conv1d_cuda_cases(input_data: list[int], weight_data: list[int], expected: list[int]):
    """Parameterized test for 1D convolution comparing CUDA and Simple backends."""
    input_cuda = Tensor.make(np.array(input_data), (1, 1, len(input_data)), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, len(weight_data)), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, len(input_data)), backend=simple_backend)
    weight_simple = Tensor.make(np.array(weight_data), (1, 1, len(weight_data)), backend=simple_backend)

    # Compute outputs
    output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)
    output_simple = minitorch.Conv1dFun.apply(input_simple, weight_simple)

    # Compare storages
    for i in range(output_cuda.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])

    # Optional: Assert expected output (if deterministic)
    assert list(output_cuda._tensor._storage) == expected


# @pytest.mark.task4_4b
# def test_conv1d_large_input_cuda_simple():
#     """Test for 1D convolution with large input comparing CUDA and Simple backend storages."""
#     input_data = [1] * 1024
#     weight_data = [1, -1]

#     # Create tensors
#     input_cuda = Tensor.make(np.array(input_data), (1, 1, 1024), backend=cuda_backend)
#     weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 2), backend=cuda_backend)

#     input_simple = Tensor.make(np.array(input_data), (1, 1, 1024), backend=simple_backend)
#     weight_simple = Tensor.make(np.array(weight_data), (1, 1, 2), backend=simple_backend)

#     # Compute outputs
#     output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)
#     output_simple = minitorch.Conv1dFun.apply(input_simple, weight_simple)

#     # Compare storages
#     for i in range(output_cuda.size):
#         assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])


# @given(
#     input_data=lists(small_floats, min_size=10, max_size=50)
# )
# @settings(max_examples=50)
# @pytest.mark.task4_4b
# def test_conv1d_randomized_cuda_simple(input_data: List[float]): 
#     """Randomized test for 1D convolution comparing CUDA and Simple backend storages."""
#     weight_data = [1, -1, 2]
#     if len(weight_data) > len(input_data):
#         return  # Skip invalid cases

#     # Create tensors
#     input_cuda = Tensor.make(np.array(input_data), (1, 1, len(input_data)), backend=cuda_backend)
#     weight_cuda = Tensor.make(np.array(weight_data), (1, 1, len(weight_data)), backend=cuda_backend)

#     input_simple = Tensor.make(np.array(input_data), (1, 1, len(input_data)), backend=simple_backend)
#     weight_simple = Tensor.make(np.array(weight_data), (1, 1, len(weight_data)), backend=simple_backend)

#     # Compute outputs
#     output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)
#     output_simple = minitorch.Conv1dFun.apply(input_simple, weight_simple)

#     # Compare storages
#     for i in range(output_cuda.size):
#         assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])

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
