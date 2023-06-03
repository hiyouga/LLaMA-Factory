# coding=utf-8
# Exports the fine-tuned model.
# Usage: python export_model.py --checkpoint_dir path_to_checkpoint --output_dir path_to_save_model


from utils import load_pretrained, prepare_args


def main():

    model_args, _, training_args, finetuning_args = prepare_args(stage="sft")
    model, tokenizer = load_pretrained(model_args, finetuning_args)
    model.save_pretrained(training_args.output_dir, max_shard_size="10GB")
    tokenizer.save_pretrained(training_args.output_dir)
    print("model and tokenizer have been saved at:", training_args.output_dir)


if __name__ == "__main__":
    main()
