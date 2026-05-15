"""Tests for configurable embedding backend/model support.

Covers:
  - Default settings preserve sentence-transformers + all-MiniLM-L6-v2
  - Unknown embedding backend raises a clear error
  - Centroid metadata matching model/backend loads successfully
  - Centroid metadata backend mismatch fails closed
  - Centroid metadata model mismatch fails closed
  - Centroid dimension mismatch fails closed
  - Missing metadata in custom centroid dir fails closed
  - build-centroids writes .npy files and centroid_metadata.json
  - OllamaEmbeddingEncoder calls /api/embed and returns ndarray

Ollama HTTP is mocked throughout — no live Ollama required.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 16  # small fake embedding dimension for fast tests


def _fake_centroids(dim: int = _DIM):
    """Return normalized (simple, complex) centroid ndarrays."""
    rng = np.random.default_rng(42)
    s = rng.random(dim).astype(np.float32)
    c = rng.random(dim).astype(np.float32)
    return s / np.linalg.norm(s), c / np.linalg.norm(c)


def _write_centroids(directory: Path, dim: int = _DIM, metadata: dict | None = None):
    """Write simple/complex .npy files and optional centroid_metadata.json."""
    s, c = _fake_centroids(dim)
    np.save(str(directory / "simple_centroid.npy"), s)
    np.save(str(directory / "complex_centroid.npy"), c)
    if metadata is not None:
        (directory / "centroid_metadata.json").write_text(json.dumps(metadata))


def _default_meta(
    backend="sentence-transformers",
    model="all-MiniLM-L6-v2",
    dim=_DIM,
) -> dict:
    return {
        "schema_version": 1,
        "embedding_backend": backend,
        "embedding_model": model,
        "dimension": dim,
        "simple_count": 5,
        "complex_count": 5,
        "created_at": "2026-01-01T00:00:00+00:00",
        "prototypes_hash": "sha256:abc123",
    }


# ---------------------------------------------------------------------------
# Settings tests
# ---------------------------------------------------------------------------


class TestEmbeddingSettings:
    def test_default_backend(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_EMBEDDING_BACKEND", raising=False)
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.EMBEDDING_BACKEND == "sentence-transformers"

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_EMBEDDING_MODEL", raising=False)
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.EMBEDDING_MODEL == "all-MiniLM-L6-v2"

    def test_default_centroid_dir_is_none(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_CENTROID_DIR", raising=False)
        from nadirclaw.settings import Settings
        s = Settings()
        assert s.CENTROID_DIR is None

    def test_custom_backend_via_env(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_EMBEDDING_BACKEND", "ollama")
        from nadirclaw.settings import Settings
        assert Settings().EMBEDDING_BACKEND == "ollama"

    def test_custom_model_via_env(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_EMBEDDING_MODEL", "nomic-embed-text")
        from nadirclaw.settings import Settings
        assert Settings().EMBEDDING_MODEL == "nomic-embed-text"

    def test_centroid_dir_via_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NADIRCLAW_CENTROID_DIR", str(tmp_path))
        from nadirclaw.settings import Settings
        assert Settings().CENTROID_DIR == tmp_path

    def test_embedding_api_base_defaults_to_ollama_api_base(self, monkeypatch):
        monkeypatch.delenv("NADIRCLAW_EMBEDDING_API_BASE", raising=False)
        monkeypatch.setenv("OLLAMA_API_BASE", "http://myhost:11434")
        from nadirclaw.settings import Settings
        assert Settings().EMBEDDING_API_BASE == "http://myhost:11434"

    def test_embedding_api_base_override(self, monkeypatch):
        monkeypatch.setenv("NADIRCLAW_EMBEDDING_API_BASE", "http://custom:9999")
        from nadirclaw.settings import Settings
        assert Settings().EMBEDDING_API_BASE == "http://custom:9999"


# ---------------------------------------------------------------------------
# Encoder factory tests
# ---------------------------------------------------------------------------


class TestEncoderFactory:
    def test_unknown_backend_raises_value_error(self):
        from nadirclaw.encoder import _build_encoder
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            _build_encoder("bogus-backend", "any-model", "http://localhost:11434")

    def test_sentence_transformers_backend_key(self):
        from nadirclaw.encoder import SentenceTransformerEncoder
        assert SentenceTransformerEncoder.backend == "sentence-transformers"

    def test_ollama_backend_key(self):
        from nadirclaw.encoder import OllamaEmbeddingEncoder
        assert OllamaEmbeddingEncoder.backend == "ollama"

    def test_build_encoder_returns_sentence_transformer(self):
        """_build_encoder with sentence-transformers should return the right type."""
        from nadirclaw.encoder import SentenceTransformerEncoder, _build_encoder

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros((_DIM,), dtype=np.float32)

        with patch("nadirclaw.encoder.SentenceTransformerEncoder.__init__", return_value=None):
            enc = _build_encoder("sentence-transformers", "all-MiniLM-L6-v2", "")
            assert isinstance(enc, SentenceTransformerEncoder)

    def test_build_encoder_returns_ollama(self):
        from nadirclaw.encoder import OllamaEmbeddingEncoder, _build_encoder
        enc = _build_encoder("ollama", "nomic-embed-text", "http://localhost:11434")
        assert isinstance(enc, OllamaEmbeddingEncoder)
        assert enc.model_id == "nomic-embed-text"

    def test_reset_shared_encoder(self, monkeypatch):
        """_reset_shared_encoder should clear the singleton."""
        from nadirclaw import encoder as enc_mod
        enc_mod._shared_encoder = MagicMock()  # set a fake singleton
        from nadirclaw.encoder import _reset_shared_encoder
        _reset_shared_encoder()
        assert enc_mod._shared_encoder is None


# ---------------------------------------------------------------------------
# OllamaEmbeddingEncoder HTTP mocking tests
# ---------------------------------------------------------------------------


class TestOllamaEncoder:
    def _make_response(self, embeddings: list[list[float]]) -> bytes:
        return json.dumps({"embeddings": embeddings}).encode()

    def test_encode_returns_ndarray(self):
        from nadirclaw.encoder import OllamaEmbeddingEncoder

        enc = OllamaEmbeddingEncoder("nomic-embed-text", "http://localhost:11434")
        fake_embs = [[0.1] * _DIM, [0.2] * _DIM]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self._make_response(fake_embs)
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = enc.encode(["hello", "world"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, _DIM)

    def test_encode_sets_dimension(self):
        from nadirclaw.encoder import OllamaEmbeddingEncoder

        enc = OllamaEmbeddingEncoder("nomic-embed-text", "http://localhost:11434")
        assert enc.dimension is None  # unknown before first encode

        fake_embs = [[0.1] * _DIM]
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = self._make_response(fake_embs)
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            enc.encode(["hello"])

        assert enc.dimension == _DIM

    def test_encode_raises_on_connection_error(self):
        import urllib.error

        from nadirclaw.encoder import OllamaEmbeddingEncoder

        enc = OllamaEmbeddingEncoder("nomic-embed-text", "http://localhost:11434")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
                enc.encode(["hello"])

    def test_encode_raises_on_empty_embeddings(self):
        from nadirclaw.encoder import OllamaEmbeddingEncoder

        enc = OllamaEmbeddingEncoder("nomic-embed-text", "http://localhost:11434")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({"embeddings": []}).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp
            with pytest.raises(RuntimeError, match="no embeddings"):
                enc.encode(["hello"])


# ---------------------------------------------------------------------------
# Centroid loading and validation tests
# ---------------------------------------------------------------------------


class TestCentroidLoading:
    """Tests for BinaryComplexityClassifier._load_centroids metadata validation."""

    def _make_settings(self, monkeypatch, backend="sentence-transformers",
                       model="all-MiniLM-L6-v2", centroid_dir=None):
        monkeypatch.setenv("NADIRCLAW_EMBEDDING_BACKEND", backend)
        monkeypatch.setenv("NADIRCLAW_EMBEDDING_MODEL", model)
        if centroid_dir is not None:
            monkeypatch.setenv("NADIRCLAW_CENTROID_DIR", str(centroid_dir))
        else:
            monkeypatch.delenv("NADIRCLAW_CENTROID_DIR", raising=False)
        from nadirclaw.settings import Settings
        return Settings()

    def test_load_centroids_with_matching_metadata(self, monkeypatch, tmp_path):
        """Centroids + matching metadata should load without error."""
        meta = _default_meta()
        _write_centroids(tmp_path, dim=_DIM, metadata=meta)
        settings = self._make_settings(monkeypatch, centroid_dir=tmp_path)

        from nadirclaw.classifier import BinaryComplexityClassifier

        simple, complex_ = BinaryComplexityClassifier._load_centroids(
            encoder=None, settings=settings
        )
        assert simple.shape == (_DIM,)
        assert complex_.shape == (_DIM,)

    def test_load_centroids_backend_mismatch_raises(self, monkeypatch, tmp_path):
        """Centroids built for ollama but current backend is sentence-transformers → fail."""
        meta = _default_meta(backend="ollama", model="nomic-embed-text", dim=_DIM)
        _write_centroids(tmp_path, dim=_DIM, metadata=meta)
        settings = self._make_settings(
            monkeypatch,
            backend="sentence-transformers",
            model="all-MiniLM-L6-v2",
            centroid_dir=tmp_path,
        )

        from nadirclaw.classifier import BinaryComplexityClassifier

        with pytest.raises(RuntimeError, match="backend"):
            BinaryComplexityClassifier._load_centroids(encoder=None, settings=settings)

    def test_load_centroids_model_mismatch_raises(self, monkeypatch, tmp_path):
        """Centroids for nomic-embed-text but current model is all-MiniLM → fail."""
        meta = _default_meta(
            backend="sentence-transformers", model="nomic-embed-text", dim=_DIM
        )
        _write_centroids(tmp_path, dim=_DIM, metadata=meta)
        settings = self._make_settings(
            monkeypatch,
            backend="sentence-transformers",
            model="all-MiniLM-L6-v2",
            centroid_dir=tmp_path,
        )

        from nadirclaw.classifier import BinaryComplexityClassifier

        with pytest.raises(RuntimeError, match="model"):
            BinaryComplexityClassifier._load_centroids(encoder=None, settings=settings)

    def test_load_centroids_dimension_mismatch_in_metadata_raises(self, monkeypatch, tmp_path):
        """Metadata dimension != actual centroid file dimension → fail."""
        meta = _default_meta(dim=999)  # wrong dimension in metadata
        _write_centroids(tmp_path, dim=_DIM, metadata=meta)  # actual dim = _DIM
        settings = self._make_settings(monkeypatch, centroid_dir=tmp_path)

        from nadirclaw.classifier import BinaryComplexityClassifier

        with pytest.raises(RuntimeError, match="dimension"):
            BinaryComplexityClassifier._load_centroids(encoder=None, settings=settings)

    def test_load_centroids_simple_complex_shape_mismatch_raises(self, monkeypatch, tmp_path):
        """simple_centroid and complex_centroid of different shapes → fail."""
        rng = np.random.default_rng(1)
        np.save(str(tmp_path / "simple_centroid.npy"), rng.random(8).astype(np.float32))
        np.save(str(tmp_path / "complex_centroid.npy"), rng.random(16).astype(np.float32))
        settings = self._make_settings(monkeypatch, centroid_dir=tmp_path)

        from nadirclaw.classifier import BinaryComplexityClassifier

        with pytest.raises(ValueError, match="dimension mismatch"):
            BinaryComplexityClassifier._load_centroids(encoder=None, settings=settings)

    def test_missing_metadata_in_custom_dir_raises(self, monkeypatch, tmp_path):
        """Custom CENTROID_DIR without metadata file should fail closed."""
        # Write centroids but NO metadata
        _write_centroids(tmp_path, dim=_DIM, metadata=None)
        settings = self._make_settings(monkeypatch, centroid_dir=tmp_path)

        from nadirclaw.classifier import BinaryComplexityClassifier

        with pytest.raises(FileNotFoundError, match="centroid_metadata.json"):
            BinaryComplexityClassifier._load_centroids(encoder=None, settings=settings)

    def test_missing_centroids_in_custom_dir_raises(self, monkeypatch, tmp_path):
        """Custom CENTROID_DIR with no .npy files → clear error."""
        settings = self._make_settings(monkeypatch, centroid_dir=tmp_path)

        from nadirclaw.classifier import BinaryComplexityClassifier

        with pytest.raises(FileNotFoundError):
            BinaryComplexityClassifier._load_centroids(encoder=None, settings=settings)

    def test_encoder_dimension_mismatch_raises(self, monkeypatch, tmp_path):
        """Encoder dimension != centroid dimension → fail."""
        meta = _default_meta(dim=_DIM)
        _write_centroids(tmp_path, dim=_DIM, metadata=meta)
        settings = self._make_settings(monkeypatch, centroid_dir=tmp_path)

        # Mock an encoder that says dimension = 999
        mock_encoder = MagicMock()
        mock_encoder.dimension = 999

        from nadirclaw.classifier import BinaryComplexityClassifier

        with pytest.raises(RuntimeError, match="Encoder output dimension"):
            BinaryComplexityClassifier._load_centroids(
                encoder=mock_encoder, settings=settings
            )

    def test_default_pkg_dir_no_metadata_loads_ok(self, monkeypatch):
        """Package directory without metadata should load silently (backward compat)."""
        monkeypatch.delenv("NADIRCLAW_CENTROID_DIR", raising=False)
        monkeypatch.setenv("NADIRCLAW_EMBEDDING_BACKEND", "sentence-transformers")
        monkeypatch.setenv("NADIRCLAW_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        from nadirclaw.settings import Settings
        settings = Settings()

        # The package .npy files are present; this should not raise even if no metadata
        from nadirclaw.classifier import BinaryComplexityClassifier

        simple, complex_ = BinaryComplexityClassifier._load_centroids(
            encoder=None, settings=settings
        )
        assert simple.ndim == 1
        assert complex_.ndim == 1
        assert simple.shape == complex_.shape


# ---------------------------------------------------------------------------
# build-centroids CLI tests
# ---------------------------------------------------------------------------


class TestBuildCentroidsCLI:
    """Tests for the build-centroids CLI command."""

    def test_build_centroids_default_writes_npy_and_metadata(self, tmp_path, monkeypatch):
        """build-centroids should write simple/complex .npy and centroid_metadata.json."""
        from click.testing import CliRunner

        from nadirclaw.cli import main

        monkeypatch.delenv("NADIRCLAW_EMBEDDING_BACKEND", raising=False)
        monkeypatch.delenv("NADIRCLAW_EMBEDDING_MODEL", raising=False)

        fake_embs = np.random.default_rng(0).random((5, _DIM)).astype(np.float32)

        with patch("nadirclaw.encoder.SentenceTransformerEncoder.__init__", return_value=None):
            with patch(
                "nadirclaw.encoder.SentenceTransformerEncoder.encode",
                return_value=fake_embs,
            ):
                with patch(
                    "nadirclaw.encoder.SentenceTransformerEncoder.dimension",
                    new_callable=lambda: property(lambda self: _DIM),
                ):
                    from nadirclaw.encoder import _reset_shared_encoder
                    _reset_shared_encoder()

                    runner = CliRunner()
                    result = runner.invoke(
                        main,
                        ["build-centroids", "--output-dir", str(tmp_path)],
                        catch_exceptions=False,
                    )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "simple_centroid.npy").exists()
        assert (tmp_path / "complex_centroid.npy").exists()
        assert (tmp_path / "centroid_metadata.json").exists()

        meta = json.loads((tmp_path / "centroid_metadata.json").read_text())
        assert meta["schema_version"] == 1
        assert meta["embedding_backend"] == "sentence-transformers"
        assert meta["embedding_model"] == "all-MiniLM-L6-v2"
        assert meta["dimension"] == _DIM
        assert "created_at" in meta
        assert "prototypes_hash" in meta

    def test_build_centroids_custom_backend_flags(self, tmp_path, monkeypatch):
        """CLI flags should override backend/model in written metadata."""
        from click.testing import CliRunner

        from nadirclaw.cli import main

        fake_embs = np.random.default_rng(1).random((5, 768)).astype(np.float32)
        fake_response = json.dumps(
            {"embeddings": fake_embs.tolist()}
        ).encode()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = fake_response
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            from nadirclaw.encoder import _reset_shared_encoder
            _reset_shared_encoder()

            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "build-centroids",
                    "--embedding-backend", "ollama",
                    "--embedding-model", "nomic-embed-text",
                    "--embedding-api-base", "http://127.0.0.1:11434",
                    "--output-dir", str(tmp_path),
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        meta = json.loads((tmp_path / "centroid_metadata.json").read_text())
        assert meta["embedding_backend"] == "ollama"
        assert meta["embedding_model"] == "nomic-embed-text"
        assert meta["dimension"] == 768

    def test_build_centroids_unknown_backend_fails(self, tmp_path, monkeypatch):
        """build-centroids with an unsupported backend should exit non-zero."""
        from click.testing import CliRunner

        from nadirclaw.cli import main
        from nadirclaw.encoder import _reset_shared_encoder
        _reset_shared_encoder()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "build-centroids",
                "--embedding-backend", "not-a-real-backend",
                "--output-dir", str(tmp_path),
            ],
        )
        assert result.exit_code != 0
