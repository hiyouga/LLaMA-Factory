import sys
sys.path.append("./")
import os
from modelscope import snapshot_download

def download_model(modelpath="LLM-Research/Meta-Llama-3-8B"):
    model_dir = snapshot_download(modelpath)
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta-llama")
    os.rename(os.path.dirname(model_dir), target_dir)


def start():
    from src import train_web_gradio
    demo = train_web_gradio.main()

if __name__ == "__main__":
    download_model()
    start()
