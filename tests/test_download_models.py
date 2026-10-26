import hashlib
import io
from unittest.mock import patch

import pytest

from tools.download_models import download


def test_download_checks_hash_reuses_verified_file_and_preserves_custom_model(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"release-model")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "models" / "upper.pt"
    download(source.as_uri(), target, expected)
    assert target.read_bytes() == source.read_bytes()
    with patch("tools.download_models.urlopen", side_effect=AssertionError("Already installed")):
        download(source.as_uri(), target, expected)
    target.write_bytes(b"custom-trained-model")
    with pytest.raises(ValueError, match="Move it aside"):
        download(source.as_uri(), target, expected)
    assert target.read_bytes() == b"custom-trained-model"


def test_corrupt_download_is_not_installed_and_partial_is_removed(tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"corrupt")
    target = tmp_path / "models" / "upper.pt"
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        download(source.as_uri(), target, "0" * 64)
    assert not target.exists()
    assert not list(target.parent.glob("*.partial"))


def test_interrupted_download_removes_partial_file(tmp_path):
    class Interrupted(io.BytesIO):
        def read(self, *_):
            raise OSError("Connection interrupted")

    target = tmp_path / "upper.pt"
    with patch("tools.download_models.urlopen", return_value=Interrupted()):
        with pytest.raises(OSError, match="interrupted"):
            download("https://github.com/example/upper.pt", target, "0" * 64)
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))
