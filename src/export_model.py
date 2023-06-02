# coding=utf-8
# Exports the fine-tuned model.
# Usage: python export_model.py --checkpoint_dir path_to_checkpoint --output_dir path_to_save_model


from transformers import HfArgumentParser, TrainingArguments
from utils import ModelArguments, FinetuningArguments, load_pretrained


def main():

    parser = HfArgumentParser((ModelArguments, TrainingArguments, FinetuningArguments))
    model_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_pretrained(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size="1GB")
    tokenizer.save_pretrained(training_args.output_dir)

    print("model and tokenizer have been saved at:", training_args.output_dir)


if __name__ == "__main__":
    main()
