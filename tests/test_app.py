"""
Lightweight CI tests — zero ML imports.
All tests are self-contained and only use the standard library.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_project_structure():
    """Test that critical project files and directories exist."""
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "app")), "app/ missing"
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "src")), "src/ missing"
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "config")), "config/ missing"
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "src", "models")), "src/models/ missing"
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "src", "inference")), "src/inference/ missing"
    assert os.path.isdir(os.path.join(PROJECT_ROOT, "src", "utils")), "src/utils/ missing"
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "app", "server.py")), "server.py missing"
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "requirements.txt")), "requirements.txt missing"


def test_requirements_file_has_core_deps():
    """Test that requirements.txt lists critical dependencies."""
    with open(os.path.join(PROJECT_ROOT, "requirements.txt"), encoding="utf-8") as f:
        content = f.read().lower()
    assert "flask" in content
    assert "torch" in content
    assert "replicate" in content
    assert "pyyaml" in content


def test_config_yaml_exists_and_valid():
    """Test that a YAML config file exists and is parseable."""
    import yaml
    config_dir = os.path.join(PROJECT_ROOT, "config")
    yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yml', '.yaml'))]
    assert len(yaml_files) > 0, "No YAML config files found in config/"
    for yf in yaml_files:
        with open(os.path.join(config_dir, yf), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None, f"{yf} is empty or invalid"


def test_server_py_has_required_routes():
    """Test that server.py defines the expected API routes."""
    with open(os.path.join(PROJECT_ROOT, "app", "server.py"), encoding="utf-8") as f:
        content = f.read()
    assert "/api/health" in content, "health endpoint missing"
    assert "/api/generate" in content, "generate endpoint missing"
    assert "/api/models" in content, "models endpoint missing"
    assert "/api/gallery" in content, "gallery endpoint missing"


def test_no_secrets_in_codebase():
    """Ensure no API keys are accidentally committed in Python files."""
    for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, "src")):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", "venv")]
        for fname in files:
            if fname.endswith(".py"):
                with open(os.path.join(root, fname), encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                assert "sk-" not in content, f"Possible API key found in {fname}"
                assert "REPLICATE_API_TOKEN =" not in content or "os.getenv" in content, \
                    f"Hardcoded token in {fname}"


def test_hf_deploy_keeps_gallery_assets():
    """Deployment workflow should prepackage gallery assets for HF Spaces."""
    workflow_path = os.path.join(PROJECT_ROOT, ".github", "workflows", "ci-cd.yml")
    with open(workflow_path, encoding="utf-8") as f:
        content = f.read()

    assert "scripts/build_gallery_manifest.py" in content
    assert "app/gallery_seed_manifest.json" in content
    assert "rm -rf data outputs/gallery outputs/logs docs" in content


def test_gallery_manifest_script_exists():
    """Gallery packaging helper should exist for deployment."""
    manifest_script = os.path.join(PROJECT_ROOT, "scripts", "build_gallery_manifest.py")
    assert os.path.isfile(manifest_script), "Gallery manifest builder is missing"


def test_server_py_has_seed_gallery_fallback():
    """Server should support gallery JSON fallback when binaries are absent."""
    with open(os.path.join(PROJECT_ROOT, "app", "server.py"), encoding="utf-8") as f:
        content = f.read()
    assert "gallery_seed_manifest.json" in content
    assert "_load_seed_gallery" in content
