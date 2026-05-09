import pytest
import os
import sys

# Ensure the app module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test that the health endpoint returns a 200 OK status."""
    rv = client.get('/api/health')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert json_data['status'] == 'ok'

def test_classes_endpoint(client):
    """Test that the classes endpoint returns successfully."""
    rv = client.get('/api/classes')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'classes' in json_data
    assert 'total' in json_data
