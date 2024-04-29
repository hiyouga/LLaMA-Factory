import os
import sys

# get the parent path of this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# add the parent path into sys
sys.path.append(current_dir)
from llmtuner import create_ui


def main():
    create_ui().queue().launch(server_name="0.0.0.0", server_port=None, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
