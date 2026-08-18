from llamafactory.extras.constants import SUPPORTED_MODELS, DownloadSource


def test_minimax_models_are_registered() -> None:
    expected_models = {
        "MiniMax-M2.7-Thinking": "MiniMaxAI/MiniMax-M2.7",
        "MiniMax-M3-Thinking": "MiniMaxAI/MiniMax-M3",
    }

    for model_name, model_path in expected_models.items():
        assert SUPPORTED_MODELS[model_name][DownloadSource.DEFAULT] == model_path
        assert SUPPORTED_MODELS[model_name][DownloadSource.MODELSCOPE] == model_path
