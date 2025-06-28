"""Shared pytest fixtures and configuration for USRNet tests."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import numpy as np
import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Return a mock configuration dictionary for testing."""
    return {
        "scale_factor": 4,
        "noise_level": 2.55,
        "kernel_width": 0.6,
        "blur_type": "gaussian_iso",
        "batch_size": 1,
        "device": "cpu",
        "model_path": None,
    }


@pytest.fixture
def sample_image_array() -> np.ndarray:
    """Create a sample image array for testing."""
    # Create a simple 64x64 RGB image
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample PyTorch tensor for testing."""
    # Create a batch of 1 RGB image with shape (1, 3, 64, 64)
    tensor = torch.randn(1, 3, 64, 64)
    return tensor


@pytest.fixture
def sample_kernel() -> np.ndarray:
    """Create a sample blur kernel for testing."""
    # Create a simple 5x5 Gaussian-like kernel
    kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float32)
    kernel = kernel / kernel.sum()  # Normalize
    return kernel


@pytest.fixture
def mock_model_weights(temp_dir: Path) -> Path:
    """Create a mock model weights file."""
    weights_path = temp_dir / "test_model.pth"
    # Create a simple state dict
    state_dict = {
        "model_state": {"layer1.weight": torch.randn(64, 3, 3, 3)},
        "optimizer_state": {},
        "epoch": 100,
    }
    torch.save(state_dict, weights_path)
    return weights_path


@pytest.fixture
def device() -> torch.device:
    """Return the appropriate PyTorch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Optional: Skip slow tests unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle slow tests."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )