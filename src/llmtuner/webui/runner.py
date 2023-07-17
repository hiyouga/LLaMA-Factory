import logging
import os
import threading
import time
import transformers
from typing import Optional, Tuple

from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.constants import DEFAULT_MODULE # will be deprecated
from llmtuner.extras.logging import LoggerHandler
from llmtuner.extras.misc import torch_gc
from llmtuner.tuner import get_train_args, run_sft
from llmtuner.webui.common import get_model_path, get_save_dir
from llmtuner.webui.locales import ALERTS
from llmtuner.webui.utils import format_info, get_eval_results


class Runner:

    def __init__(self):
        self.aborted = False
        self.running = False

    def set_abort(self):
        self.aborted = True
        self.running = False

    def initialize(self, lang: str, model_name: str, dataset: list) -> Tuple[str, str, LoggerHandler, LogCallback]:
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

    def finalize(self, lang: str, finish_info: Optional[str] = None) -> str:
        self.running = False
        torch_gc()
        if self.aborted:
            return ALERTS["info_aborted"][lang]
        else:
            return finish_info if finish_info is not None else ALERTS["info_finished"][lang]

    def run_train(
        self, lang, model_name, checkpoints, finetuning_type, template,
        dataset, dataset_dir, learning_rate, num_train_epochs, max_samples,
        fp16, quantization_bit, batch_size, gradient_accumulation_steps,
        lr_scheduler_type, logging_steps, save_steps, output_dir
    ):
        model_name_or_path, error, logger_handler, trainer_callback = self.initialize(lang, model_name, dataset)
        if error:
            yield error
            return

        if checkpoints:
            checkpoint_dir = ",".join(
                [os.path.join(get_save_dir(model_name), finetuning_type, checkpoint) for checkpoint in checkpoints]
            )
        else:
            checkpoint_dir = None

        args = dict(
            model_name_or_path=model_name_or_path,
            do_train=True,
            finetuning_type=finetuning_type,
            lora_target=DEFAULT_MODULE.get(model_name.split("-")[0], None) or "q_proj,v_proj",
            prompt_template=template,
            dataset=",".join(dataset),
            dataset_dir=dataset_dir,
            max_samples=int(max_samples),
            output_dir=os.path.join(get_save_dir(model_name), finetuning_type, output_dir),
            checkpoint_dir=checkpoint_dir,
            overwrite_cache=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            save_steps=save_steps,
            learning_rate=float(learning_rate),
            num_train_epochs=float(num_train_epochs),
            fp16=fp16,
            quantization_bit=int(quantization_bit) if quantization_bit else None
        )
        model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)

        run_args = dict(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            callbacks=[trainer_callback]
        )
        thread = threading.Thread(target=run_sft, kwargs=run_args)
        thread.start()

        while thread.is_alive():
            time.sleep(1)
            if self.aborted:
                yield ALERTS["info_aborting"][lang]
            else:
                yield format_info(logger_handler.log, trainer_callback.tracker)

        yield self.finalize(lang)

    def run_eval(
        self, lang, model_name, checkpoints, finetuning_type, template,
        dataset, dataset_dir, max_samples, batch_size, quantization_bit, predict
    ):
        model_name_or_path, error, logger_handler, trainer_callback = self.initialize(lang, model_name, dataset)
        if error:
            yield error
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
            model_name_or_path=model_name_or_path,
            do_eval=True,
            finetuning_type=finetuning_type,
            prompt_template=template,
            dataset=",".join(dataset),
            dataset_dir=dataset_dir,
            max_samples=int(max_samples),
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            overwrite_cache=True,
            predict_with_generate=True,
            per_device_eval_batch_size=batch_size,
            quantization_bit=int(quantization_bit) if quantization_bit else None
        )

        if predict:
            args.pop("do_eval", None)
            args["do_predict"] = True

        model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)

        run_args = dict(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            callbacks=[trainer_callback]
        )
        thread = threading.Thread(target=run_sft, kwargs=run_args)
        thread.start()

        while thread.is_alive():
            time.sleep(1)
            if self.aborted:
                yield ALERTS["info_aborting"][lang]
            else:
                yield format_info(logger_handler.log, trainer_callback.tracker)

        yield self.finalize(lang, get_eval_results(os.path.join(output_dir, "all_results.json")))
