import os
import time
import logging
import gradio as gr
from threading import Thread
from gradio.components import Component # cannot use TYPE_CHECKING here
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Tuple

import transformers
from transformers.trainer import TRAINING_ARGS_NAME

from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.constants import TRAINING_STAGES
from llmtuner.extras.logging import LoggerHandler
from llmtuner.extras.misc import torch_gc
from llmtuner.tuner import run_exp
from llmtuner.webui.common import get_module, get_save_dir, load_config
from llmtuner.webui.locales import ALERTS
from llmtuner.webui.utils import gen_cmd, get_eval_results, update_process_bar

if TYPE_CHECKING:
    from llmtuner.webui.manager import Manager


class Runner:

    def __init__(self, manager: "Manager") -> None:
        self.manager = manager
        self.thread: "Thread" = None
        self.data: Dict["Component", Any] = None
        self.do_train = True
        self.monitor_inputs: Dict[str, str] = None
        self.aborted = False
        self.running = False
        self.logger_handler = LoggerHandler()
        self.logger_handler.setLevel(logging.INFO)
        logging.root.addHandler(self.logger_handler)
        transformers.logging.add_handler(self.logger_handler)

    @property
    def alive(self) -> bool:
        return self.thread is not None

    def set_abort(self) -> None:
        self.aborted = True
        self.running = False

    def _initialize(self, lang: str, model_name: str, model_path: str, dataset: List[str]) -> str:
        if self.running:
            return ALERTS["err_conflict"][lang]

        if not model_name:
            return ALERTS["err_no_model"][lang]

        if not model_path:
            return ALERTS["err_no_path"][lang]

        if len(dataset) == 0:
            return ALERTS["err_no_dataset"][lang]

        self.aborted = False
        self.logger_handler.reset()
        self.trainer_callback = LogCallback(self)
        return ""

    def _finalize(self, lang: str, finish_info: str) -> str:
        self.thread = None
        self.running = False
        torch_gc()
        if self.aborted:
            return ALERTS["info_aborted"][lang]
        else:
            return finish_info

    def _parse_train_args(self, data: Dict[Component, Any]) -> Tuple[str, str, str, List[str], str, Dict[str, Any]]:
        get = lambda name: data[self.manager.get_elem(name)]
        user_config = load_config()

        if get("top.checkpoints"):
            checkpoint_dir = ",".join([
                get_save_dir(get("top.model_name"), get("top.finetuning_type"), ckpt) for ckpt in get("top.checkpoints")
            ])
        else:
            checkpoint_dir = None

        output_dir = get_save_dir(get("top.model_name"), get("top.finetuning_type"), get("train.output_dir"))

        args = dict(
            stage=TRAINING_STAGES[get("train.training_stage")],
            model_name_or_path=get("top.model_path"),
            do_train=True,
            overwrite_cache=False,
            cache_dir=user_config.get("cache_dir", None),
            checkpoint_dir=checkpoint_dir,
            finetuning_type=get("top.finetuning_type"),
            quantization_bit=int(get("top.quantization_bit")) if get("top.quantization_bit") in ["8", "4"] else None,
            template=get("top.template"),
            system_prompt=get("top.system_prompt"),
            flash_attn=get("top.flash_attn"),
            shift_attn=get("top.shift_attn"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") in ["linear", "dynamic"] else None,
            dataset_dir=get("train.dataset_dir"),
            dataset=",".join(get("train.dataset")),
            cutoff_len=get("train.cutoff_len"),
            learning_rate=float(get("train.learning_rate")),
            num_train_epochs=float(get("train.num_train_epochs")),
            max_samples=int(get("train.max_samples")),
            per_device_train_batch_size=get("train.batch_size"),
            gradient_accumulation_steps=get("train.gradient_accumulation_steps"),
            lr_scheduler_type=get("train.lr_scheduler_type"),
            max_grad_norm=float(get("train.max_grad_norm")),
            logging_steps=get("train.logging_steps"),
            save_steps=get("train.save_steps"),
            warmup_steps=get("train.warmup_steps"),
            lora_rank=get("train.lora_rank"),
            lora_dropout=get("train.lora_dropout"),
            lora_target=get("train.lora_target") or get_module(get("top.model_name")),
            resume_lora_training=get("train.resume_lora_training"),
            output_dir=output_dir
        )
        args[get("train.compute_type")] = True
        args["disable_tqdm"] = True

        if TRAINING_STAGES[get("train.training_stage")] in ["rm", "ppo", "dpo"]:
            args["resume_lora_training"] = (args["quantization_bit"] is not None)

        if args["quantization_bit"] is not None:
            args["upcast_layernorm"] = True

        if args["stage"] == "ppo":
            args["reward_model"] = get("train.reward_model")

        if args["stage"] == "dpo":
            args["dpo_beta"] = get("train.dpo_beta")

        if get("train.val_size") > 1e-6 and args["stage"] != "ppo":
            args["val_size"] = get("train.val_size")
            args["evaluation_strategy"] = "steps"
            args["eval_steps"] = get("train.save_steps")
            args["load_best_model_at_end"] = True

        return get("top.lang"), get("top.model_name"), get("top.model_path"), get("train.dataset"), output_dir, args

    def _parse_eval_args(self, data: Dict[Component, Any]) -> Tuple[str, str, str, List[str], str, Dict[str, Any]]:
        get = lambda name: data[self.manager.get_elem(name)]
        user_config = load_config()

        if get("top.checkpoints"):
            checkpoint_dir = ",".join([
                get_save_dir(get("top.model_name"), get("top.finetuning_type"), ckpt) for ckpt in get("top.checkpoints")
            ])
            output_dir = get_save_dir(
                get("top.model_name"), get("top.finetuning_type"), "eval_" + "_".join(get("top.checkpoints"))
            )
        else:
            checkpoint_dir = None
            output_dir = get_save_dir(get("top.model_name"), get("top.finetuning_type"), "eval_base")

        args = dict(
            stage="sft",
            model_name_or_path=get("top.model_path"),
            do_eval=True,
            overwrite_cache=False,
            predict_with_generate=True,
            cache_dir=user_config.get("cache_dir", None),
            checkpoint_dir=checkpoint_dir,
            finetuning_type=get("top.finetuning_type"),
            quantization_bit=int(get("top.quantization_bit")) if get("top.quantization_bit") in ["8", "4"] else None,
            template=get("top.template"),
            system_prompt=get("top.system_prompt"),
            flash_attn=get("top.flash_attn"),
            shift_attn=get("top.shift_attn"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") in ["linear", "dynamic"] else None,
            dataset_dir=get("eval.dataset_dir"),
            dataset=",".join(get("eval.dataset")),
            cutoff_len=get("eval.cutoff_len"),
            max_samples=int(get("eval.max_samples")),
            per_device_eval_batch_size=get("eval.batch_size"),
            max_new_tokens=get("eval.max_new_tokens"),
            top_p=get("eval.top_p"),
            temperature=get("eval.temperature"),
            output_dir=get("eval.output_dir")
        )

        if get("eval.predict"):
            args.pop("do_eval", None)
            args["do_predict"] = True

        return get("top.lang"), get("top.model_name"), get("top.model_path"), get("eval.dataset"), output_dir, args

    def _preview(self, data: Dict[Component, Any], do_train: bool) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        parse_func = self._parse_train_args if do_train else self._parse_eval_args
        lang, model_name, model_path, dataset, _, args = parse_func(data)
        error = self._initialize(lang, model_name, model_path, dataset)
        if error:
            yield error, gr.update(visible=False)
        else:
            yield gen_cmd(args), gr.update(visible=False)

    def _launch(self, data: Dict[Component, Any], do_train: bool) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        parse_func = self._parse_train_args if do_train else self._parse_eval_args
        lang, model_name, model_path, dataset, output_dir, args = parse_func(data)
        self.data, self.do_train, self.monitor_inputs = data, do_train, dict(lang=lang, output_dir=output_dir)
        error = self._initialize(lang, model_name, model_path, dataset)
        if error:
            yield error, gr.update(visible=False)
        else:
            self.running = True
            run_kwargs = dict(args=args, callbacks=[self.trainer_callback])
            self.thread = Thread(target=run_exp, kwargs=run_kwargs)
            self.thread.start()
            yield from self.monitor()

    def preview_train(self, data: Dict[Component, Any]) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._preview(data, do_train=True)

    def preview_eval(self, data: Dict[Component, Any]) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._preview(data, do_train=False)

    def run_train(self, data: Dict[Component, Any]) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._launch(data, do_train=True)

    def run_eval(self, data: Dict[Component, Any]) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        yield from self._launch(data, do_train=False)

    def monitor(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        lang, output_dir = self.monitor_inputs["lang"], self.monitor_inputs["output_dir"]
        while self.thread.is_alive():
            time.sleep(2)
            if self.aborted:
                yield ALERTS["info_aborting"][lang], gr.update(visible=False)
            else:
                yield self.logger_handler.log, update_process_bar(self.trainer_callback)

        if self.do_train:
            if os.path.exists(os.path.join(output_dir, TRAINING_ARGS_NAME)):
                finish_info = ALERTS["info_finished"][lang]
            else:
                finish_info = ALERTS["err_failed"][lang]
        else:
            if os.path.exists(os.path.join(output_dir, "all_results.json")):
                finish_info = get_eval_results(os.path.join(output_dir, "all_results.json"))
            else:
                finish_info = ALERTS["err_failed"][lang]

        yield self._finalize(lang, finish_info), gr.update(visible=False)
