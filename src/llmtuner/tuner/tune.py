import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core import get_train_args, load_model_and_tokenizer
from llmtuner.tuner.pt import run_pt
from llmtuner.tuner.sft import run_sft
from llmtuner.tuner.rm import run_rm
from llmtuner.tuner.ppo import run_ppo
from llmtuner.tuner.dpo import run_dpo
from llmtuner.dsets import get_dataset, preprocess_dataset

if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if general_args.stage == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "rm":
        run_rm(model_args, data_args, training_args, finetuning_args, callbacks)
    elif general_args.stage == "ppo":
        run_ppo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif general_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, callbacks)
    else:
        raise ValueError("Unknown task.")


def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, data_args, training_args, finetuning_args, _, _ = get_train_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    if model_args.gptq_export:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        dataset = get_dataset(model_args, data_args)
        dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")
        dataset_for_gptq = []
        for index in range(len(dataset["input_ids"])):
            # remove eos_id
            dataset_for_gptq.append(
                {
                    "input_ids": dataset["input_ids"][index][:-1],
                    "attention_mask": dataset["attention_mask"][index][:-1]
                }
            )
        quantize_config = BaseQuantizeConfig(
            bits=model_args.gptq_quantization_bit,
            group_size=128,
            desc_act=False,
        )

        tmp_path = os.path.join(training_args.output_dir, "unquantified")
        # for custom model, AutoGPTQ package only accept model_path as input
        model.save_pretrained(tmp_path, max_shard_size=max_shard_size)
        # load merged model locally
        gptq_model = AutoGPTQForCausalLM.from_pretrained(tmp_path, quantize_config)
        # remove unquantified model
        shutil.rmtree(tmp_path)
        logger.info("start quantization")
        gptq_model.quantize(dataset_for_gptq)
        logger.info("save gptq model")
        gptq_model.save_quantized(training_args.output_dir)
    else:
        model.save_pretrained(training_args.output_dir, max_shard_size=max_shard_size)

    try:
        tokenizer.save_pretrained(training_args.output_dir)
    except:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()
