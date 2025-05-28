import pytest
pd = pytest.importorskip("pandas")

from faster_llm.ML.keras_model import KerasModel
from faster_llm.ML.pytorch_model import PyTorchModel
from faster_llm.ML.pytorch_lightning_model import PyTorchLightningModel


class DummyModel:
    def fit(self, X, y, epochs=1, verbose=0):
        self.fitted = True

    def predict(self, X):
        return [0 for _ in range(len(X))]


def _run_wrapper(wrapper_cls):
    recorded = {}

    def fake_send(message, server_url=None):
        recorded["msg"] = message
        recorded["url"] = server_url

    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    y = pd.Series([0, 0, 0, 0])
    wrapper_cls = wrapper_cls  # rename

    from faster_llm import LLM
    original = LLM.send_to_llm
    LLM.send_to_llm = fake_send
    try:
        wrapper_cls(DummyModel(), df, y, send_to_llm_flag=True, server_url="http://host")
    finally:
        LLM.send_to_llm = original

    assert recorded["url"] == "http://host"
    assert "accuracy" in recorded["msg"]


def test_keras_wrapper():
    _run_wrapper(KerasModel)


def test_pytorch_wrapper():
    _run_wrapper(PyTorchModel)


def test_pytorch_lightning_wrapper():
    _run_wrapper(PyTorchLightningModel)
