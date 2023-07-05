import os
import sys
import torch
import hashlib
from itertools import chain
from typing import List, Literal, Optional, Tuple

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import datasets
from datasets import Dataset, concatenate_datasets, load_dataset

from peft import (
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model
)

from peft.utils import CONFIG_NAME, WEIGHTS_NAME

from trl import AutoModelForCausalLMWithValueHead

from .config import (
    ModelArguments,
    DataTrainingArguments,
    FinetuningArguments,
    GeneratingArguments
)

from .template import Template

from .other import (
    get_logger,
    load_trainable_params,
    load_valuehead_params,
    print_trainable_params,
    prepare_model_for_training,
    IGNORE_INDEX
)

check_min_version("4.29.1")
require_version("datasets>=2.12.0", "To fix: pip install datasets>=2.12.0")
require_version("accelerate>=0.19.0", "To fix: pip install accelerate>=0.19.0")
require_version("peft>=0.3.0", "To fix: pip install peft>=0.3.0")
require_version("trl>=0.4.4", "To fix: pip install trl>=0.4.4")


logger = get_logger(__name__)


def _init_adapter(
        model: PreTrainedModel,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool,
        is_mergeable: bool
) -> PreTrainedModel:
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full":
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        logger.info("Fine-tuning method: Freeze")

        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

        if model_args.checkpoint_dir is not None:
            assert load_trainable_params(model, model_args.checkpoint_dir[0]), "Model checkpoint is not correctly loaded."

    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)), \
                "Provided path ({}) does not contain a LoRA weight.".format(model_args.checkpoint_dir[0])
            assert os.path.exists(os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)), \
                "The given checkpoint may be not a LoRA checkpoint, please specify `--finetuning_type full/freeze` instead."

            if (is_trainable and model_args.resume_lora_training) or (not is_mergeable): # continually train on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                logger.info("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if lastest_checkpoint is not None: # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=is_trainable)

        if is_trainable and lastest_checkpoint is None: # create new lora weights while training
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        logger.info("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


def load_pretrained(
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: Optional[bool] = False,
        stage: Optional[Literal["pt", "sft", "rm", "ppo"]] = "sft"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """
    if (not is_trainable) and model_args.checkpoint_dir is None:
        logger.warning("Checkpoint is not found at evaluation, load the original model.")
        finetuning_args = FinetuningArguments(finetuning_type="none")

    assert stage in ["pt", "sft"] or finetuning_args.finetuning_type == "lora", \
        "RM and PPO training can only be performed with the LoRA method."

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
        **config_kwargs
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == 64000: # 64000 for baichuan model (older version)
        tokenizer.pad_token_id = 0 # set as the <unk> token

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    is_mergeable = True

    # Quantization configurations (using bitsandbytes library).
    if model_args.quantization_bit is not None:
        if model_args.quantization_bit == 8:
            require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )

        elif model_args.quantization_bit == 4:
            require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
            require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
            require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
            require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_args.compute_dtype,
                bnb_4bit_use_double_quant=model_args.double_quantization,
                bnb_4bit_quant_type=model_args.quantization_type
            )

        is_mergeable = False
        config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        logger.info("Quantizing model to {} bit.".format(model_args.quantization_bit))

    if not is_trainable: # `device_map=auto` should be used for inference only
        config_kwargs["device_map"] = "auto"

    if model_args.checkpoint_dir is not None and finetuning_args.finetuning_type == "full":
        model_to_load = model_args.checkpoint_dir[0]
    else:
        model_to_load = model_args.model_name_or_path

    # Load and prepare pretrained models (without valuehead).
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        config=config,
        torch_dtype=torch.bfloat16 if model_args.compute_dtype == torch.bfloat16 else torch.float16,
        low_cpu_mem_usage=True,
        **config_kwargs
    )

    # Register auto class to save the custom code files.
    if hasattr(config, "auto_map") and "AutoConfig" in config.auto_map:
        config.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoTokenizer" in config.auto_map:
        tokenizer.__class__.register_for_auto_class()
    if hasattr(config, "auto_map") and "AutoModelForCausalLM" in config.auto_map:
        model.__class__.register_for_auto_class()

    # Initialize adapters
    model = prepare_model_for_training(model, finetuning_args.finetuning_type) if is_trainable else model
    model = _init_adapter(model, model_args, finetuning_args, is_trainable, is_mergeable)

    if stage == "rm" or stage == "ppo": # add value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        if stage == "rm" and model_args.checkpoint_dir is not None: # load valuehead weights to evaluate reward model
            logger.warning("Only the last checkpoint containing valuehead will be loaded as the valuehead.")
            if load_valuehead_params(model, model_args.checkpoint_dir[-1]):
                model.v_head.load_state_dict({
                    "summary.weight": getattr(model, "reward_head_weight"),
                    "summary.bias": getattr(model, "reward_head_bias")
                })

        if stage == "ppo": # load reward model
            assert is_trainable, "PPO stage cannot be performed at evaluation."
            assert model_args.reward_model is not None, "Reward model is necessary for PPO training."
            logger.info("Load reward model from {}".format(model_args.reward_model))
            model.pretrained_model.load_adapter(model_args.reward_model, "reward", is_trainable=False)
            assert load_valuehead_params(model, model_args.reward_model), "Reward model is not correctly loaded."

    if not is_trainable:
        model.requires_grad_(False) # fix all model params
        model = model.half() if model_args.quantization_bit is None else model # cast from fp32 to fp16

    print_trainable_params(model)

    return model, tokenizer


def prepare_args(
        stage: Literal["pt", "sft", "rm", "ppo"]
) -> Tuple[ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, FinetuningArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, data_args, training_args, finetuning_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, finetuning_args = parser.parse_args_into_dataclasses()

    # Setup logging
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Check arguments (do not check finetuning_args since it may be loaded from checkpoints)
    data_args.init_for_training()

    assert stage == "sft" or (not training_args.predict_with_generate), \
        "`predict_with_generate` cannot be set as True at PT, RM and PPO stages."

    assert not (training_args.do_train and training_args.predict_with_generate), \
        "`predict_with_generate` cannot be set as True while training."

    assert (not training_args.do_predict) or training_args.predict_with_generate, \
        "Please enable `predict_with_generate` to save model predictions."

    assert model_args.quantization_bit is None or finetuning_args.finetuning_type == "lora", \
        "Quantization is only compatible with the LoRA method."

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            assert len(model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."
        else:
            assert model_args.quantization_bit is None or len(model_args.checkpoint_dir) == 1, \
                "Quantized model only accepts a single checkpoint."

    if model_args.quantization_bit is not None and (not training_args.do_train):
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if training_args.do_train and (not training_args.fp16):
        logger.warning("We recommend enable fp16 mixed precision training.")

    if data_args.prompt_template == "default":
        logger.warning("Please specify `prompt_template` if you are using other pre-trained models.")

    if training_args.local_rank != -1 and training_args.ddp_find_unused_parameters is None:
        logger.warning("`ddp_find_unused_parameters` needs to be set as False in DDP training.")
        training_args.ddp_find_unused_parameters = False

    training_args.optim = "adamw_torch" if training_args.optim == "adamw_hf" else training_args.optim # suppress warning

    if model_args.quantization_bit is not None:
        if training_args.fp16:
            model_args.compute_dtype = torch.float16
        elif training_args.bf16:
            model_args.compute_dtype = torch.bfloat16
        else:
            model_args.compute_dtype = torch.float32

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}\n"
        + f"  distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, finetuning_args


def prepare_infer_args() -> Tuple[ModelArguments, DataTrainingArguments, FinetuningArguments, GeneratingArguments]:

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuningArguments, GeneratingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"): # Provide arguments with a json file.
        model_args, data_args, finetuning_args, generating_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, finetuning_args, generating_args = parser.parse_args_into_dataclasses()

    assert model_args.quantization_bit is None or finetuning_args.finetuning_type == "lora", \
        "Quantization is only compatible with the LoRA method."

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            assert len(model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."
        else:
            assert model_args.quantization_bit is None or len(model_args.checkpoint_dir) == 1, \
                "Quantized model only accepts a single checkpoint."

    if data_args.prompt_template == "default":
        logger.warning("Please specify `prompt_template` if you are using other pre-trained models.")

    return model_args, data_args, finetuning_args, generating_args


def prepare_data(
        model_args: ModelArguments,
        data_args: DataTrainingArguments
) -> Dataset:

    def checksum(file_path, hash):
        with open(file_path, "rb") as datafile:
            binary_data = datafile.read()
        sha1 = hashlib.sha1(binary_data).hexdigest()
        if sha1 != hash:
            logger.warning("Checksum failed for {}. It may vary depending on the platform.".format(file_path))

    ext2type = {
        "csv": "csv",
        "json": "json",
        "jsonl": "json",
        "txt": "text"
    }

    max_samples = data_args.max_samples
    all_datasets: List[Dataset] = [] # support multiple datasets

    for dataset_attr in data_args.dataset_list:

        logger.info("Loading dataset {}...".format(dataset_attr))

        if dataset_attr.load_from == "hf_hub":
            data_path = dataset_attr.dataset_name
            data_files = None
        elif dataset_attr.load_from == "script":
            data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
            data_files = None
        elif dataset_attr.load_from == "file":
            data_path = None
            data_files: List[str] = []

            if os.path.isdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                for file_name in os.listdir(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                    data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name, file_name))

                    if data_path is None:
                        data_path = ext2type.get(data_files[0].split(".")[-1], None)
                    else:
                        assert data_path == ext2type.get(data_files[-1].split(".")[-1], None), "file type does not match."
            elif os.path.isfile(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)):
                data_files.append(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name))
                data_path = ext2type.get(data_files[0].split(".")[-1], None)
            else:
                raise ValueError("File not found.")

            assert data_path, "File extension must be txt, csv, json or jsonl."

            if len(data_files) == 1 and dataset_attr.dataset_sha1 is not None:
                checksum(data_files[0], dataset_attr.dataset_sha1)
            else:
                logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json or too many files.")
        else:
            raise NotImplementedError

        raw_datasets = load_dataset(
            data_path,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None
        )
        dataset = raw_datasets[data_args.split]

        if max_samples is not None:
            max_samples_temp = min(len(dataset), max_samples)
            dataset = dataset.select(range(max_samples_temp))

        dummy_data = [None] * len(dataset)
        prefix_data = [dataset_attr.source_prefix] * len(dataset)
        for column_name, target_name in [
            ("prompt_column", "prompt"),
            ("query_column", "query"),
            ("response_column", "response"),
            ("history_column", "history")
        ]: # every dataset will have 4 columns same as each other
            if getattr(dataset_attr, column_name) != target_name:
                if getattr(dataset_attr, column_name):
                    dataset = dataset.rename_column(getattr(dataset_attr, column_name), target_name)
                else: # None or empty string
                    dataset = dataset.add_column(target_name, dummy_data)
        dataset = dataset.add_column("prefix", prefix_data)
        all_datasets.append(dataset)

    if len(data_args.dataset_list) == 1:
        all_datasets = all_datasets[0]
    else:
        all_datasets = concatenate_datasets(all_datasets)

    return all_datasets


def preprocess_data(
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_args: DataTrainingArguments,
        training_args: Seq2SeqTrainingArguments,
        stage: Literal["pt", "sft", "rm", "ppo"]
) -> Dataset:

    column_names = list(dataset.column_names)
    prompt_template = Template(data_args.prompt_template)

    # support question with a single answer or multiple answers
    def get_dialog(examples):
        for i in range(len(examples["prompt"])):
            if examples["prompt"][i] and examples["response"][i]:
                query, answer = examples["prompt"][i], examples["response"][i]
                query = query + "\n" + examples["query"][i] if examples["query"][i] else query
                prefix = examples["prefix"][i] if examples["prefix"][i] else ""
                dialog = prompt_template.get_dialog(query, answer, examples["history"][i], prefix)
                yield dialog

    def preprocess_pretrain_dataset(examples):
        # build grouped texts with format `<bos> X1 X2 X3 ...` (without <eos>)
        text_ids = tokenizer(examples["prompt"], add_special_tokens=False)["input_ids"]
        concatenated_ids = list(chain(*text_ids))
        total_length = len(concatenated_ids)
        block_size = data_args.max_source_length - 1
        # we drop the small remainder, and if the total_length < block_size, we exclude this batch
        total_length = (total_length // block_size) * block_size
        # split by chunks of max_source_length
        result = [[tokenizer.bos_token_id] + concatenated_ids[i: i + block_size]
                  for i in range(0, total_length, block_size)]
        return {
            "input_ids": result,
            "labels": result.copy()
        }

    def preprocess_supervised_dataset(examples):
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for input with history, we build multiple input-label pairs just like:
        # https://github.com/lm-sys/FastChat/blob/f17c092f64840fa6354ed52789dccb2daa793d0b/fastchat/train/train.py#L112
        model_inputs = {"input_ids": [], "labels": []}
        max_length = data_args.max_source_length + data_args.max_target_length

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2*i], add_special_tokens=True)
                target_ids = tokenizer.encode(text=dialog[2*i+1], add_special_tokens=False)

                if len(source_ids) > data_args.max_source_length:
                    source_ids = source_ids[:data_args.max_source_length]
                if len(target_ids) > data_args.max_target_length - 1: # eos token
                    target_ids = target_ids[:data_args.max_target_length - 1]

                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]
                labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

        return model_inputs

    def preprocess_unsupervised_dataset(examples):
        # build inputs with format `<bos> X` and labels with format `<bos> Y`
        model_inputs = {"input_ids": [], "labels": []}

        for dialog in get_dialog(examples):
            prompt, answer = "".join(dialog[:-1]), dialog[-1]

            source_ids = tokenizer.encode(text=prompt, add_special_tokens=True)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=True)

            if len(source_ids) > data_args.max_source_length:
                source_ids = source_ids[:data_args.max_source_length]
            if len(target_ids) > data_args.max_target_length:
                target_ids = target_ids[:data_args.max_target_length]

            model_inputs["input_ids"].append(source_ids)
            model_inputs["labels"].append(target_ids)

        return model_inputs

    def preprocess_pairwise_dataset(examples):
        # build input pairs with format `<bos> X Y1 <eos>` and `<bos> X Y2 <eos>`
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for dialog in get_dialog(examples):
            prompt, answer = "".join(dialog[:-1]), dialog[-1]

            source_ids = tokenizer.encode(text=prompt, add_special_tokens=True)
            accept_ids = tokenizer.encode(text=answer[0], add_special_tokens=False)
            reject_ids = tokenizer.encode(text=answer[1], add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length:
                source_ids = source_ids[:data_args.max_source_length]
            if len(accept_ids) > data_args.max_target_length - 1: # eos token
                accept_ids = accept_ids[:data_args.max_target_length - 1]
            if len(reject_ids) > data_args.max_target_length - 1: # eos token
                reject_ids = reject_ids[:data_args.max_target_length - 1]

            accept_ids = source_ids + accept_ids + [tokenizer.eos_token_id]
            reject_ids = source_ids + reject_ids + [tokenizer.eos_token_id]

            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    def print_pairwise_dataset_example(example):
        print("accept_ids:\n{}".format(example["accept_ids"]))
        print("accepts:\n{}".format(tokenizer.decode(example["accept_ids"], skip_special_tokens=False)))
        print("reject_ids:\n{}".format(example["reject_ids"]))
        print("rejects:\n{}".format(tokenizer.decode(example["reject_ids"], skip_special_tokens=False)))

    def print_unsupervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

    if stage == "pt":
        preprocess_function = preprocess_pretrain_dataset
    elif stage == "sft":
        preprocess_function = preprocess_unsupervised_dataset \
            if training_args.predict_with_generate else preprocess_supervised_dataset
    elif stage == "rm":
        preprocess_function = preprocess_pairwise_dataset
    elif stage == "ppo":
        preprocess_function = preprocess_unsupervised_dataset

    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset"
        )

        if stage == "pt":
            print_unsupervised_dataset_example(dataset[0])
        elif stage == "sft":
            print_supervised_dataset_example(dataset[0])
        elif stage == "rm":
            print_pairwise_dataset_example(dataset[0])
        elif stage == "ppo":
            print_unsupervised_dataset_example(dataset[0])

        return dataset
