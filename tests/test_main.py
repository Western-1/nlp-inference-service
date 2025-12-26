import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_redirect():
    """
    New test: Verify that root redirects to documentation.
    """
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"

def test_health_check_connected():
    """
    Fixed: Now points to /health instead of /
    """
    with patch("app.main.get_redis") as mock_get_redis:
        mock_r = mock_get_redis.return_value
        mock_r.ping.return_value = True

        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Online & Monitored with W&B"
        assert data["db_status"] == "Connected to Redis"

def test_get_history_empty():
    with patch("app.main.get_redis") as mock_get_redis:
        mock_r = mock_get_redis.return_value
        mock_r.lrange.return_value = []
        
        response = client.get("/history")
        assert response.status_code == 200
        assert response.json() == []

@patch("app.main.get_model")
@patch("app.main.get_redis")
def test_sentiment_analysis_mocked(mock_get_redis, mock_get_model):
    mock_r = mock_get_redis.return_value
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [{"label": "POSITIVE", "score": 0.99}]
    
    mock_get_model.return_value = mock_pipeline

    payload = {"text": "MLOps is amazing!"}
    response = client.post("/sentiment", json=payload)

    assert response.status_code == 200
    data = response.json()
    
    assert data["result"][0]["label"] == "POSITIVE"
    assert data["result"][0]["score"] == 0.99

    mock_r.lpush.assert_called_once()
    mock_r.ltrim.assert_called_once()

def test_input_validation_too_long():
    long_text = "a" * 1001
    payload = {"text": long_text}
    response = client.post("/sentiment", json=payload)
    assert response.status_code == 422