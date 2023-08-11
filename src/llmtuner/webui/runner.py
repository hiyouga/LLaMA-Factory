import gradio as gr
import logging
import os
import threading
import time
import transformers
from transformers.trainer import TRAINING_ARGS_NAME
from typing import Generator, List, Tuple

from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.constants import DEFAULT_MODULE
from llmtuner.extras.logging import LoggerHandler
from llmtuner.extras.misc import torch_gc
from llmtuner.tuner import run_exp
from llmtuner.webui.common import get_model_path, get_save_dir
from llmtuner.webui.locales import ALERTS
from llmtuner.webui.utils import get_eval_results, update_process_bar


class Runner:

    def __init__(self):
        self.aborted = False
        self.running = False

    def set_abort(self):
        self.aborted = True
        self.running = False

    def initialize(
        self, lang: str, model_name: str, dataset: List[str]
    ) -> Tuple[str, str, LoggerHandler, LogCallback]:
        if self.running:
            return None, ALERTS["err_conflict"][lang], None, None

        if not model_name:
            return None, ALERTS["err_no_model"][lang], None, None

        model_name_or_path = get_model_path(model_name)
        if not model_name_or_path:
            return None, ALERTS["err_no_path"][lang], None, None

        if len(dataset) == 0:
            return None, ALERTS["err_no_dataset"][lang], None, None

        self.aborted = False
        self.running = True

        logger_handler = LoggerHandler()
        logger_handler.setLevel(logging.INFO)
        logging.root.addHandler(logger_handler)
        transformers.logging.add_handler(logger_handler)
        trainer_callback = LogCallback(self)

        return model_name_or_path, "", logger_handler, trainer_callback

    def finalize(
        self, lang: str, finish_info: str
    ) -> str:
        self.running = False
        torch_gc()
        if self.aborted:
            return ALERTS["info_aborted"][lang]
        else:
            return finish_info

    def run_train(
        self,
        lang: str,
        model_name: str,
        checkpoints: List[str],
        finetuning_type: str,
        quantization_bit: str,
        template: str,
        source_prefix: str,
        dataset_dir: str,
        dataset: List[str],
        max_source_length: int,
        max_target_length: int,
        learning_rate: str,
        num_train_epochs: str,
        max_samples: str,
        batch_size: int,
        gradient_accumulation_steps: int,
        lr_scheduler_type: str,
        max_grad_norm: str,
        val_size: float,
        logging_steps: int,
        save_steps: int,
        warmup_steps: int,
        compute_type: str,
        padding_side: str,
        lora_rank: int,
        lora_dropout: float,
        lora_target: str,
        resume_lora_training: bool,
        output_dir: str
    ) -> Generator[str, None, None]:
        model_name_or_path, error, logger_handler, trainer_callback = self.initialize(lang, model_name, dataset)
        if error:
            yield error, gr.update(visible=False)
            return

        if checkpoints:
            checkpoint_dir = ",".join(
                [os.path.join(get_save_dir(model_name), finetuning_type, checkpoint) for checkpoint in checkpoints]
            )
        else:
            checkpoint_dir = None

        output_dir = os.path.join(get_save_dir(model_name), finetuning_type, output_dir)

        args = dict(
            stage="sft",
            model_name_or_path=model_name_or_path,
            do_train=True,
            overwrite_cache=True,
            checkpoint_dir=checkpoint_dir,
            finetuning_type=finetuning_type,
            quantization_bit=int(quantization_bit) if quantization_bit else None,
            template=template,
            source_prefix=source_prefix,
            dataset_dir=dataset_dir,
            dataset=",".join(dataset),
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            learning_rate=float(learning_rate),
            num_train_epochs=float(num_train_epochs),
            max_samples=int(max_samples),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=float(max_grad_norm),
            logging_steps=logging_steps,
            save_steps=save_steps,
            warmup_steps=warmup_steps,
            fp16=(compute_type == "fp16"),
            bf16=(compute_type == "bf16"),
            padding_side=padding_side,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            lora_target=lora_target or DEFAULT_MODULE.get(model_name.split("-")[0], "q_proj,v_proj"),
            resume_lora_training=resume_lora_training,
            output_dir=output_dir
        )

        if val_size > 1e-6:
            args["val_size"] = val_size
            args["evaluation_strategy"] = "steps"
            args["eval_steps"] = save_steps
            args["load_best_model_at_end"] = True

        run_kwargs = dict(args=args, callbacks=[trainer_callback])
        thread = threading.Thread(target=run_exp, kwargs=run_kwargs)
        thread.start()

        while thread.is_alive():
            time.sleep(2)
            if self.aborted:
                yield ALERTS["info_aborting"][lang], gr.update(visible=False)
            else:
                yield logger_handler.log, update_process_bar(trainer_callback)

        if os.path.exists(os.path.join(output_dir, TRAINING_ARGS_NAME)):
            finish_info = ALERTS["info_finished"][lang]
        else:
            finish_info = ALERTS["err_failed"][lang]

        yield self.finalize(lang, finish_info), gr.update(visible=False)

    def run_eval(
        self,
        lang: str,
        model_name: str,
        checkpoints: List[str],
        finetuning_type: str,
        quantization_bit: str,
        template: str,
        source_prefix: str,
        dataset_dir: str,
        dataset: List[str],
        max_source_length: int,
        max_target_length: int,
        max_samples: str,
        batch_size: int,
        predict: bool
    ) -> Generator[str, None, None]:
        model_name_or_path, error, logger_handler, trainer_callback = self.initialize(lang, model_name, dataset)
        if error:
            yield error, gr.update(visible=False)
            return

        if checkpoints:
            checkpoint_dir = ",".join(
                [os.path.join(get_save_dir(model_name), finetuning_type, checkpoint) for checkpoint in checkpoints]
            )
            output_dir = os.path.join(get_save_dir(model_name), finetuning_type, "eval_" + "_".join(checkpoints))
        else:
            checkpoint_dir = None
            output_dir = os.path.join(get_save_dir(model_name), finetuning_type, "eval_base")

        args = dict(
            stage="sft",
            model_name_or_path=model_name_or_path,
            do_eval=True,
            overwrite_cache=True,
            predict_with_generate=True,
            checkpoint_dir=checkpoint_dir,
            finetuning_type=finetuning_type,
            quantization_bit=int(quantization_bit) if quantization_bit else None,
            template=template,
            source_prefix=source_prefix,
            dataset_dir=dataset_dir,
            dataset=",".join(dataset),
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            max_samples=int(max_samples),
            per_device_eval_batch_size=batch_size,
            output_dir=output_dir
        )

        if predict:
            args.pop("do_eval", None)
            args["do_predict"] = True

        run_kwargs = dict(args=args, callbacks=[trainer_callback])
        thread = threading.Thread(target=run_exp, kwargs=run_kwargs)
        thread.start()

        while thread.is_alive():
            time.sleep(2)
            if self.aborted:
                yield ALERTS["info_aborting"][lang], gr.update(visible=False)
            else:
                yield logger_handler.log, update_process_bar(trainer_callback)

        if os.path.exists(os.path.join(output_dir, "all_results.json")):
            finish_info = get_eval_results(os.path.join(output_dir, "all_results.json"))
        else:
            finish_info = ALERTS["err_failed"][lang]

        yield self.finalize(lang, finish_info), gr.update(visible=False)
