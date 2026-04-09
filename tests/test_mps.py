"""tests for apple mps device support."""

from unittest.mock import MagicMock, patch

from autiobook.tts import TTSConfig, empty_device_cache, get_default_device, is_mps


class TestDeviceDetection:
    """tests for device auto-detection."""

    @patch("autiobook.tts.torch")
    def test_mps_detected_when_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.hip = None
        mock_torch.backends.mps.is_available.return_value = True
        assert get_default_device() == "mps"

    @patch("autiobook.tts.torch")
    def test_cuda_preferred_over_mps(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.hip = None
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.backends.mps.is_available.return_value = True
        assert get_default_device() == "cuda"

    @patch("autiobook.tts.torch")
    def test_cpu_fallback_when_no_gpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.version.hip = None
        mock_torch.backends.mps.is_available.return_value = False
        assert get_default_device() == "cpu"


class TestIsMps:
    """tests for is_mps helper."""

    def test_explicit_mps_device(self):
        assert is_mps("mps") is True

    def test_explicit_cuda_device(self):
        assert is_mps("cuda") is False

    def test_explicit_cpu_device(self):
        assert is_mps("cpu") is False


class TestEmptyDeviceCache:
    """tests for device-aware cache clearing."""

    @patch("autiobook.tts.torch")
    def test_cuda_cache_cleared(self, mock_torch):
        empty_device_cache("cuda")
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("autiobook.tts.torch")
    def test_cuda_device_index_cache_cleared(self, mock_torch):
        empty_device_cache("cuda:0")
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("autiobook.tts.torch")
    def test_mps_cache_cleared(self, mock_torch):
        empty_device_cache("mps")
        mock_torch.mps.empty_cache.assert_called_once()

    @patch("autiobook.tts.torch")
    def test_cpu_no_cache_cleared(self, mock_torch):
        empty_device_cache("cpu")
        mock_torch.cuda.empty_cache.assert_not_called()
        mock_torch.mps.empty_cache.assert_not_called()


class TestTTSConfigMPS:
    """tests for MPS-specific model loading behavior."""

    @patch("autiobook.tts.get_default_device", return_value="mps")
    def test_mps_config_defaults(self, _mock_device):
        config = TTSConfig()
        assert config.device == "mps"

    def test_mps_dtype_is_bfloat16(self):
        """verify MPS uses bfloat16, not float32."""
        import torch

        from autiobook.tts import TTSEngine

        config = TTSConfig(device="mps", warmup=False)
        engine = TTSEngine(config)

        # patch the model loading to inspect dtype/attn choices
        with patch("autiobook.tts.Qwen3TTSModel", create=True) as mock_cls:
            with patch.dict("sys.modules", {"qwen_tts": MagicMock(Qwen3TTSModel=mock_cls)}):
                engine._load_model()

            call_kwargs = mock_cls.from_pretrained.call_args
            assert call_kwargs.kwargs["dtype"] == torch.bfloat16
            assert call_kwargs.kwargs["attn_implementation"] == "sdpa"

    def test_cuda_dtype_is_bfloat16(self):
        """verify CUDA also uses bfloat16."""
        import torch

        from autiobook.tts import TTSEngine

        config = TTSConfig(device="cuda", warmup=False)
        engine = TTSEngine(config)

        with patch("autiobook.tts.Qwen3TTSModel", create=True) as mock_cls:
            with patch.dict("sys.modules", {"qwen_tts": MagicMock(Qwen3TTSModel=mock_cls)}):
                with patch("autiobook.tts.is_rocm", return_value=False):
                    engine._load_model()

            call_kwargs = mock_cls.from_pretrained.call_args
            assert call_kwargs.kwargs["dtype"] == torch.bfloat16
            assert call_kwargs.kwargs["attn_implementation"] == "flash_attention_2"

    def test_cpu_dtype_is_float32(self):
        """verify CPU uses float32."""
        import torch

        from autiobook.tts import TTSEngine

        config = TTSConfig(device="cpu", warmup=False)
        engine = TTSEngine(config)

        with patch("autiobook.tts.Qwen3TTSModel", create=True) as mock_cls:
            with patch.dict("sys.modules", {"qwen_tts": MagicMock(Qwen3TTSModel=mock_cls)}):
                engine._load_model()

            call_kwargs = mock_cls.from_pretrained.call_args
            assert call_kwargs.kwargs["dtype"] == torch.float32
            assert call_kwargs.kwargs["attn_implementation"] == "sdpa"

    def test_rocm_uses_sdpa(self):
        """verify ROCm uses sdpa, not flash_attention_2."""
        from autiobook.tts import TTSEngine

        config = TTSConfig(device="cuda", warmup=False)
        engine = TTSEngine(config)

        with patch("autiobook.tts.Qwen3TTSModel", create=True) as mock_cls:
            with patch.dict("sys.modules", {"qwen_tts": MagicMock(Qwen3TTSModel=mock_cls)}):
                with patch("autiobook.tts.is_rocm", return_value=True):
                    engine._load_model()

            call_kwargs = mock_cls.from_pretrained.call_args
            assert call_kwargs.kwargs["attn_implementation"] == "sdpa"
