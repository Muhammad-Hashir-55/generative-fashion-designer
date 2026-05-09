"""
Lightweight CI tests — no heavy ML imports (torch, diffusers, etc.)
These run fast on GitHub Actions runners without GPU.
"""

import os
import sys
import importlib

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_config_loads():
    """Test that the YAML config loads without error."""
    from src.utils.config import load_config
    config = load_config()
    assert config is not None
    assert config is not None


def test_dtd_classes_defined():
    """Test that DTD texture classes are defined."""
    from src.data.dataset import DTD_CLASSES
    assert isinstance(DTD_CLASSES, list)
    assert len(DTD_CLASSES) > 0


def test_replicate_generator_module_exists():
    """Test that the replicate generator module can be found."""
    spec = importlib.util.find_spec("src.inference.replicate_generator")
    assert spec is not None, "replicate_generator module not found"


def test_project_structure():
    """Test that critical project files and directories exist."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    assert os.path.isdir(os.path.join(project_root, "app")), "app/ directory missing"
    assert os.path.isdir(os.path.join(project_root, "src")), "src/ directory missing"
    assert os.path.isfile(os.path.join(project_root, "requirements.txt")), "requirements.txt missing"
    assert os.path.isfile(os.path.join(project_root, "app", "server.py")), "app/server.py missing"


def test_requirements_file_has_core_deps():
    """Test that requirements.txt lists critical dependencies."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(project_root, "requirements.txt"), encoding="utf-8") as f:
        content = f.read().lower()
    assert "flask" in content
    assert "torch" in content
    assert "replicate" in content
