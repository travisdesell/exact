import os
import tempfile
from data_collectors.macro_api import data_dir_check


def test_data_dir_check_create(monkeypatch):
    # Temporary path that doesn't exist yet
    temp_dir = tempfile.mkdtemp() + '/macro'

    # No input needed because directory doesn't exist
    result = data_dir_check(temp_dir)

    assert os.path.exists(temp_dir)
    assert result is True