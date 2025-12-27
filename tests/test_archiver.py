import json
import os
from unittest.mock import MagicMock, patch
from archiver.main import archive_logs

@patch("archiver.main.get_s3_client")
@patch("archiver.main.get_redis")
def test_archive_logs_success(mock_get_redis, mock_get_s3):
    mock_redis = mock_get_redis.return_value
    mock_s3 = mock_get_s3.return_value
    
    mock_redis.exists.return_value = True
    mock_redis.lrange.return_value = [json.dumps({"test": "log"})]
    
    with patch("builtins.open", MagicMock()), patch("os.remove"):
        archive_logs()

    mock_redis.rename.assert_called_with("api_logs", "logs_to_upload")
    mock_s3.upload_file.assert_called_once()
    mock_redis.delete.assert_called_with("logs_to_upload")

@patch("archiver.main.get_redis")
def test_archive_logs_empty_redis(mock_get_redis):
    mock_redis = mock_get_redis.return_value
    mock_redis.exists.return_value = False
    
    archive_logs()
    
    mock_redis.rename.assert_not_called()