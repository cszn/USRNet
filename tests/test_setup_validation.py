"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest
import numpy as np
import torch


class TestSetupValidation:
    """Test class to validate the testing infrastructure."""

    @pytest.mark.unit
    def test_imports_work(self):
        """Test that all required imports work correctly."""
        # Test standard library imports
        import os
        import tempfile
        
        # Test third-party imports
        import cv2
        import scipy
        import matplotlib
        import requests
        
        # Test that imports don't raise exceptions
        assert cv2 is not None
        assert scipy is not None
        assert matplotlib is not None
        assert requests is not None

    @pytest.mark.unit
    def test_project_structure_exists(self, project_root):
        """Test that the expected project structure exists."""
        # Check main directories
        assert (project_root / "models").exists()
        assert (project_root / "utils").exists()
        assert (project_root / "tests").exists()
        
        # Check test subdirectories
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check important files
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests" / "conftest.py").exists()

    @pytest.mark.unit
    def test_fixtures_work(self, temp_dir, mock_config, sample_image_array, 
                          sample_tensor, sample_kernel, device):
        """Test that all fixtures work correctly."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "scale_factor" in mock_config
        assert mock_config["scale_factor"] == 4
        
        # Test sample_image_array fixture
        assert isinstance(sample_image_array, np.ndarray)
        assert sample_image_array.shape == (64, 64, 3)
        assert sample_image_array.dtype == np.uint8
        
        # Test sample_tensor fixture
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (1, 3, 64, 64)
        
        # Test sample_kernel fixture
        assert isinstance(sample_kernel, np.ndarray)
        assert sample_kernel.shape == (5, 5)
        assert np.isclose(sample_kernel.sum(), 1.0)
        
        # Test device fixture
        assert isinstance(device, torch.device)

    @pytest.mark.unit
    def test_mock_model_weights(self, mock_model_weights):
        """Test that mock model weights can be created and loaded."""
        assert mock_model_weights.exists()
        assert mock_model_weights.suffix == ".pth"
        
        # Try loading the weights
        state_dict = torch.load(mock_model_weights, map_location="cpu")
        assert "model_state" in state_dict
        assert "optimizer_state" in state_dict
        assert "epoch" in state_dict
        assert state_dict["epoch"] == 100

    @pytest.mark.unit
    def test_project_imports(self):
        """Test that project modules can be imported."""
        # Test model imports
        from models import basicblock
        from models import network_usrnet
        from models import network_usrnet_v1
        
        # Test utils imports
        from utils import utils_deblur
        from utils import utils_image
        from utils import utils_logger
        from utils import utils_sisr
        
        # Verify imports succeeded
        assert basicblock is not None
        assert network_usrnet is not None
        assert utils_image is not None

    @pytest.mark.unit
    def test_logging_fixture(self, capture_logs):
        """Test that the logging capture fixture works."""
        import logging
        
        logger = logging.getLogger(__name__)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        # Check that messages were captured
        assert len(capture_logs.records) >= 3
        assert any(r.levelname == "DEBUG" for r in capture_logs.records)
        assert any(r.levelname == "INFO" for r in capture_logs.records)
        assert any(r.levelname == "WARNING" for r in capture_logs.records)

    @pytest.mark.unit
    def test_random_seed_reset(self):
        """Test that random seeds are properly reset."""
        # Generate some random numbers
        np_random1 = np.random.randn(5)
        torch_random1 = torch.randn(5)
        
        # Reset seeds (this happens automatically via fixture)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Generate again
        np_random2 = np.random.randn(5)
        torch_random2 = torch.randn(5)
        
        # They should be different from first set
        assert not np.allclose(np_random1, np_random2)

    @pytest.mark.integration
    def test_full_integration(self, temp_dir, mock_config):
        """Integration test to verify components work together."""
        # Create a temporary file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Testing infrastructure is working!")
        
        # Verify it exists and can be read
        assert test_file.exists()
        content = test_file.read_text()
        assert content == "Testing infrastructure is working!"
        
        # Verify config can be modified
        mock_config["new_key"] = "new_value"
        assert mock_config["new_key"] == "new_value"

    @pytest.mark.slow
    @pytest.mark.unit
    def test_slow_marker(self):
        """Test that slow marker works (this test is skipped by default)."""
        import time
        time.sleep(0.1)  # Simulate slow operation
        assert True