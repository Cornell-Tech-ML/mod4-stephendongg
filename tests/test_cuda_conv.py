import pytest
from hypothesis.strategies import floats
import minitorch
from minitorch import Tensor
import numpy as np
from .strategies import assert_close


# Define small floats for testing
small_floats = floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

cuda_backend = minitorch.TensorBackend(minitorch.CudaOps)
simple_backend = minitorch.TensorBackend(minitorch.SimpleOps)


@pytest.mark.task4_4b
def test_conv1d_cuda_simple() -> None:
    input_data = [1, 1, 1, 1]
    weight_data = [1, 1, 4]

    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 3), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4), backend=simple_backend)
    weight_simple = Tensor.make(
        np.array(weight_data), (1, 1, 3), backend=simple_backend
    )

    output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)

    # Torch convolution
    output = minitorch.Conv1dFun.apply(input_simple, weight_simple)
    print(output._tensor)
    print(output_cuda._tensor)
    print("LENGTHS")
    print(len(output._tensor._storage), len(output_cuda._tensor._storage))

    for i in range(output.size):
        assert_close(output_cuda._tensor._storage[i], output._tensor._storage[i])


@pytest.mark.task4_4b
def test_conv1d_zero_weight_cuda_simple() -> None:
    """Test for 1D convolution with all-zero weights comparing CUDA and Simple backend storages."""
    input_data = [1, 2, 3, 4]
    weight_data = [0, 0, 0]

    # Create tensors
    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 3), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4), backend=simple_backend)
    weight_simple = Tensor.make(
        np.array(weight_data), (1, 1, 3), backend=simple_backend
    )

    # Compute outputs
    output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)
    output_simple = minitorch.Conv1dFun.apply(input_simple, weight_simple)

    # Compare storages
    for i in range(output_simple.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])


@pytest.mark.task4_4b
@pytest.mark.parametrize(
    "input_data, weight_data",
    [
        # Simple Cases
        ([1, 2, 3, 4], [0, 0, 0]),  # All-zero weights
        ([1, 2, 3, 4], [1, 0, 0]),  # Single active weight
        ([1, 1, 1, 1], [1, 1, 1]),  # Uniform weights
        ([1, -1, 1, -1], [1, -1, 1]),  # Alternating values
        # Edge Cases
        ([0, 0, 0, 0], [0, 0, 0]),  # All-zero inputs and weights
        ([1, 1, 1, 1], [1, -1, 1]),  # Weights sum to 1
        ([10, 20, 30, 40], [1, 2, 3]),  # Larger inputs
    ],
)
def test_conv1d_cuda_cases(input_data: list[int], weight_data: list[int]) -> None:
    """Parameterized test for 1D convolution comparing CUDA and Simple backends."""
    input_cuda = Tensor.make(
        np.array(input_data), (1, 1, len(input_data)), backend=cuda_backend
    )
    weight_cuda = Tensor.make(
        np.array(weight_data), (1, 1, len(weight_data)), backend=cuda_backend
    )

    input_simple = Tensor.make(
        np.array(input_data), (1, 1, len(input_data)), backend=simple_backend
    )
    weight_simple = Tensor.make(
        np.array(weight_data), (1, 1, len(weight_data)), backend=simple_backend
    )

    # Compute outputs
    print("ORIGINAL SHAPE", input_cuda._tensor._shape, weight_cuda._tensor._shape)
    output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)
    output_simple = minitorch.Conv1dFun.apply(input_simple, weight_simple)

    # Compare storages
    for i in range(output_simple.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])

    print("LATER SHAPE", input_cuda._tensor._shape, weight_cuda._tensor._shape)
    # minitorch.grad_check(minitorch.Conv1dFun.apply, input_simple, output_simple)
    minitorch.grad_check(minitorch.Conv1dCudaFun.apply, input_cuda, weight_cuda)


@pytest.mark.task4_4b
def test_conv1d_simple_cuda() -> None:
    """Simple deterministic test for CUDA 1D convolution."""
    t = minitorch.tensor([0, 1, 2, 3], backend=cuda_backend).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor([[1, 2, 3]], backend=cuda_backend).view(1, 1, 3)
    out = minitorch.Conv1dCudaFun.apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@pytest.mark.task4_4b
def test_gradient() -> None:
    """Test for 1D convolution with all-zero weights comparing CUDA and Simple backend storages."""
    input_data = [1, 2, 3, 4]
    weight_data = [1, 0, 0]

    # Create tensors
    input_cuda = Tensor.make(np.array(input_data), (1, 1, 4), backend=cuda_backend)
    weight_cuda = Tensor.make(np.array(weight_data), (1, 1, 3), backend=cuda_backend)

    input_simple = Tensor.make(np.array(input_data), (1, 1, 4), backend=simple_backend)
    weight_simple = Tensor.make(
        np.array(weight_data), (1, 1, 3), backend=simple_backend
    )

    # Compute outputs
    output_cuda = minitorch.Conv1dCudaFun.apply(input_cuda, weight_cuda)
    output_simple = minitorch.Conv1dFun.apply(input_simple, weight_simple)

    # Compare storages
    for i in range(output_simple.size):
        assert_close(output_cuda._tensor._storage[i], output_simple._tensor._storage[i])

    minitorch.grad_check(minitorch.Conv1dFun.apply, input_simple, weight_simple)
    minitorch.grad_check(minitorch.Conv1dCudaFun.apply, input_cuda, weight_cuda)


@pytest.mark.task4_4b
@pytest.mark.parametrize(
    "input_data, weight_data, expected_output",
    [
        # Simple Case: Uniform Inputs and Weights
        (
            np.array(
                [[[[1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1]]]],
                dtype=np.float64,
            ),  # Input shape: (1, 2, 2, 4)
            np.array(
                [[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]], dtype=np.float64
            ),  # Weight shape: (1, 2, 2, 2)
            np.array(
                [[[[8.0, 8.0, 8.0, 4.0], [4.0, 4.0, 4.0, 2.0]]]], dtype=np.float64
            ),  # Expected Output shape: (1, 1, 2, 4)
        ),
    ],
)
def test_conv2d_cuda_cases(
    input_data: np.ndarray, weight_data: np.ndarray, expected_output: np.ndarray
) -> None:
    """Parameterized test for 2D convolution comparing CUDA and Simple backends."""

    input_flat = input_data.flatten().tolist()
    weight_flat = weight_data.flatten().tolist()
    expected_flat = expected_output.flatten().tolist()

    # Create CUDA tensors
    input_cuda = Tensor.make(
        input_flat,
        (
            input_data.shape[0],
            input_data.shape[1],
            input_data.shape[2],
            input_data.shape[3],
        ),
        backend=cuda_backend,
    )
    weight_cuda = Tensor.make(
        weight_flat,
        (
            weight_data.shape[0],
            weight_data.shape[1],
            weight_data.shape[2],
            weight_data.shape[3],
        ),
        backend=cuda_backend,
    )

    # Create Simple backend tensors
    input_simple = Tensor.make(
        input_flat,
        (
            input_data.shape[0],
            input_data.shape[1],
            input_data.shape[2],
            input_data.shape[3],
        ),
        backend=simple_backend,
    )
    weight_simple = Tensor.make(
        weight_flat,
        (
            weight_data.shape[0],
            weight_data.shape[1],
            weight_data.shape[2],
            weight_data.shape[3],
        ),
        backend=simple_backend,
    )

    # Compute forward outputs
    output_simple = minitorch.Conv2dFun.apply(input_simple, weight_simple)

    output_cuda = minitorch.Conv2dCudaFun.apply(input_cuda, weight_cuda)

    print("FLAT HERE!")
    print(expected_flat)

    # Compare forward pass outputs
    for i in range(len(expected_flat)):
        cuda_val = output_cuda._tensor._storage[i]
        simple_val = output_simple._tensor._storage[i]
        expected_val = expected_flat[i]
        print(
            f"Output[{i}]: CUDA={cuda_val}, Simple={simple_val}, Expected={expected_val}"
        )
        assert_close(cuda_val, simple_val)
        assert_close(cuda_val, expected_val)


@pytest.mark.task4_4b
@pytest.mark.parametrize(
    "input_data, weight_data, expected_output",
    [
        # Simplified Gradient Check Case: 1x1 Input and Kernel
        (
            np.array([[[[1.0]]]], dtype=np.float64),  # Input shape: (1, 1, 1, 1)
            np.array([[[[1.0]]]], dtype=np.float64),  # Weight shape: (1, 1, 1, 1)
            np.array(
                [[[[1.0]]]], dtype=np.float64
            ),  # Expected Output shape: (1, 1, 1, 1)
        ),
    ],
)
def test_conv2d_cuda_gradients(
    input_data: np.ndarray, weight_data: np.ndarray, expected_output: np.ndarray
) -> None:
    """Simplified test for 2D convolution gradient comparison between CUDA and Simple backends."""

    print("SHAPES!:")
    print(input_data.shape, " ", weight_data.shape)

    input_flat = input_data.flatten().tolist()
    weight_flat = weight_data.flatten().tolist()
    expected_flat = expected_output.flatten().tolist()

    # Create CUDA tensors with cuda_backend
    input_cuda = Tensor.make(
        input_flat,
        (
            input_data.shape[0],
            input_data.shape[1],
            input_data.shape[2],
            input_data.shape[3],
        ),
        backend=cuda_backend,
    )
    weight_cuda = Tensor.make(
        weight_flat,
        (
            weight_data.shape[0],
            weight_data.shape[1],
            weight_data.shape[2],
            weight_data.shape[3],
        ),
        backend=cuda_backend,
    )

    # Create Simple backend tensors
    input_simple = Tensor.make(
        input_flat,
        (
            input_data.shape[0],
            input_data.shape[1],
            input_data.shape[2],
            input_data.shape[3],
        ),
        backend=simple_backend,
    )
    weight_simple = Tensor.make(
        weight_flat,
        (
            weight_data.shape[0],
            weight_data.shape[1],
            weight_data.shape[2],
            weight_data.shape[3],
        ),
        backend=simple_backend,
    )

    # Compute forward outputs
    output_simple = minitorch.Conv2dFun.apply(input_simple, weight_simple)
    output_cuda = minitorch.Conv2dCudaFun.apply(input_cuda, weight_cuda)

    print("FLAT HERE!")
    print(expected_flat)

    # Compare forward pass outputs
    for i in range(len(expected_flat)):
        cuda_val = output_cuda._tensor._storage[i]
        simple_val = output_simple._tensor._storage[i]
        expected_val = expected_flat[i]
        print(
            f"Output[{i}]: CUDA={cuda_val}, Simple={simple_val}, Expected={expected_val}"
        )
        assert_close(cuda_val, simple_val)
        assert_close(cuda_val, expected_val)

    # Perform gradient check
    minitorch.grad_check(minitorch.Conv2dCudaFun.apply, input_cuda, weight_cuda)
