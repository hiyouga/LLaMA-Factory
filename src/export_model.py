# coding=utf-8
# Exports the fine-tuned model.
# Usage: python export_model.py --checkpoint_dir path_to_checkpoint --output_dir path_to_save_model

from llmtuner import export_model


def main():
    export_model()


if __name__ == "__main__":
    main()
