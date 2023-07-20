LOCALES = {
    "lang": {
        "en": {
            "label": "Lang"
        },
        "zh": {
            "label": "语言"
        }
    },
    "model_name": {
        "en": {
            "label": "Model name"
        },
        "zh": {
            "label": "模型名称"
        }
    },
    "model_path": {
        "en": {
            "label": "Model path",
            "info": "Path to pretrained model or model identifier from Hugging Face."
        },
        "zh": {
            "label": "模型路径",
            "info": "本地模型的文件路径或 Hugging Face 的模型标识符。"
        }
    },
    "finetuning_type": {
        "en": {
            "label": "Finetuning method"
        },
        "zh": {
            "label": "微调方法"
        }
    },
    "checkpoints": {
        "en": {
            "label": "Checkpoints"
        },
        "zh": {
            "label": "模型断点"
        }
    },
    "refresh_btn": {
        "en": {
            "value": "Refresh checkpoints"
        },
        "zh": {
            "value": "刷新断点"
        }
    },
    "advanced_tab": {
        "en": {
            "label": "Advanced configurations"
        },
        "zh": {
            "label": "高级设置"
        }
    },
    "quantization_bit": {
        "en": {
            "label": "Quantization bit (optional)",
            "info": "Enable 4/8-bit model quantization."
        },
        "zh": {
            "label": "量化等级（非必填）",
            "info": "启用 4/8 比特模型量化。"
        }
    },
    "template": {
        "en": {
            "label": "Prompt template",
            "info": "The template used in constructing prompts."
        },
        "zh": {
            "label": "提示模板",
            "info": "构建提示词时使用的模板"
        }
    },
    "source_prefix": {
        "en": {
            "label": "System prompt (optional)",
            "info": "A sequence used as the default system prompt."
        },
        "zh": {
            "label": "系统提示词（非必填）",
            "info": "默认使用的系统提示词"
        }
    },
    "dataset_dir": {
        "en": {
            "label": "Data dir",
            "info": "Path of the data directory."
        },
        "zh": {
            "label": "数据路径",
            "info": "数据文件夹的路径。"
        }
    },
    "dataset": {
        "en": {
            "label": "Dataset"
        },
        "zh": {
            "label": "数据集"
        }
    },
    "preview_btn": {
        "en": {
            "value": "Preview"
        },
        "zh": {
            "value": "预览"
        }
    },
    "preview_count": {
        "en": {
            "label": "Count"
        },
        "zh": {
            "label": "数量"
        }
    },
    "preview_samples": {
        "en": {
            "label": "Samples"
        },
        "zh": {
            "label": "样例"
        }
    },
    "close_btn": {
        "en": {
            "value": "Close"
        },
        "zh": {
            "value": "关闭"
        }
    },
    "max_source_length": {
        "en": {
            "label": "Max source length",
            "info": "Max tokens in source sequence."
        },
        "zh": {
            "label": "输入序列最大长度",
            "info": "输入序列分词后的最大长度。"
        }
    },
    "max_target_length": {
        "en": {
            "label": "Max target length",
            "info": "Max tokens in target sequence."
        },
        "zh": {
            "label": "输出序列最大长度",
            "info": "输出序列分词后的最大长度。"
        }
    },
    "learning_rate": {
        "en": {
            "label": "Learning rate",
            "info": "Initial learning rate for AdamW."
        },
        "zh": {
            "label": "学习率",
            "info": "AdamW 优化器的初始学习率。"
        }
    },
    "num_train_epochs": {
        "en": {
            "label": "Epochs",
            "info": "Total number of training epochs to perform."
        },
        "zh": {
            "label": "训练轮数",
            "info": "需要执行的训练总轮数。"
        }
    },
    "max_samples": {
        "en": {
            "label": "Max samples",
            "info": "Maximum samples per dataset."
        },
        "zh": {
            "label": "最大样本数",
            "info": "每个数据集最多使用的样本数。"
        }
    },
    "batch_size": {
        "en": {
            "label": "Batch size",
            "info": "Number of samples to process per GPU."
        },
        "zh":{
            "label": "批处理大小",
            "info": "每块 GPU 上处理的样本数量。"
        }
    },
    "gradient_accumulation_steps": {
        "en": {
            "label": "Gradient accumulation",
            "info": "Number of gradient accumulation steps."
        },
        "zh": {
            "label": "梯度累积",
            "info": "梯度累积的步数。"
        }
    },
    "lr_scheduler_type": {
        "en": {
            "label": "LR Scheduler",
            "info": "Name of learning rate scheduler.",
        },
        "zh": {
            "label": "学习率调节器",
            "info": "采用的学习率调节器名称。"
        }
    },
    "max_grad_norm": {
        "en": {
            "label": "Maximum gradient norm",
            "info": "Norm for gradient clipping.."
        },
        "zh": {
            "label": "最大梯度范数",
            "info": "用于梯度裁剪的范数。"
        }
    },
    "dev_ratio": {
        "en": {
            "label": "Dev ratio",
            "info": "Proportion of data in the dev set."
        },
        "zh": {
            "label": "验证集比例",
            "info": "验证集占全部样本的百分比。"
        }
    },
    "logging_steps": {
        "en": {
            "label": "Logging steps",
            "info": "Number of steps between two logs."
        },
        "zh": {
            "label": "日志间隔",
            "info": "每两次日志输出间的更新步数。"
        }
    },
    "save_steps": {
        "en": {
            "label": "Save steps",
            "info": "Number of steps between two checkpoints."
        },
        "zh": {
            "label": "保存间隔",
            "info": "每两次断点保存间的更新步数。"
        }
    },
    "warmup_steps": {
        "en": {
            "label": "Warmup steps",
            "info": "Number of steps used for warmup."
        },
        "zh": {
            "label": "预热步数",
            "info": "学习率预热采用的步数。"
        }
    },
    "compute_type": {
        "en": {
            "label": "Compute type",
            "info": "Whether to use fp16 or bf16 mixed precision training."
        },
        "zh": {
            "label": "计算类型",
            "info": "是否启用 FP16 或 BF16 混合精度训练。"
        }
    },
    "lora_tab": {
        "en": {
            "label": "LoRA configurations"
        },
        "zh": {
            "label": "LoRA 参数设置"
        }
    },
    "lora_rank": {
        "en": {
            "label": "LoRA rank",
            "info": "The rank of LoRA matrices."
        },
        "zh": {
            "label": "LoRA 秩",
            "info": "LoRA 矩阵的秩。"
        }
    },
    "lora_dropout": {
        "en": {
            "label": "LoRA Dropout",
            "info": "Dropout ratio of LoRA weights."
        },
        "zh": {
            "label": "LoRA 随机丢弃",
            "info": "LoRA 权重随机丢弃的概率。"
        }
    },
    "lora_target": {
        "en": {
            "label": "LoRA modules (optional)",
            "info": "The name(s) of target modules to apply LoRA. Use commas to separate multiple modules."
        },
        "zh": {
            "label": "LoRA 作用层（非必填）",
            "info": "应用 LoRA 的线性层名称。使用英文逗号分隔多个名称。"
        }
    },
    "start_btn": {
        "en": {
            "value": "Start"
        },
        "zh": {
            "value": "开始"
        }
    },
    "stop_btn": {
        "en": {
            "value": "Abort"
        },
        "zh": {
            "value": "中断"
        }
    },
    "output_dir": {
        "en": {
            "label": "Checkpoint name",
            "info": "Directory to save checkpoint."
        },
        "zh": {
            "label": "断点名称",
            "info": "保存模型断点的文件夹名称。"
        }
    },
    "output_box": {
        "en": {
            "value": "Ready."
        },
        "zh": {
            "value": "准备就绪。"
        }
    },
    "loss_viewer": {
        "en": {
            "label": "Loss"
        },
        "zh": {
            "label": "损失"
        }
    },
    "predict": {
        "en": {
            "label": "Save predictions"
        },
        "zh": {
            "label": "保存预测结果"
        }
    },
    "load_btn": {
        "en": {
            "value": "Load model"
        },
        "zh": {
            "value": "加载模型"
        }
    },
    "unload_btn": {
        "en": {
            "value": "Unload model"
        },
        "zh": {
            "value": "卸载模型"
        }
    },
    "info_box": {
        "en": {
            "value": "Model unloaded, please load a model first."
        },
        "zh": {
            "value": "模型未加载，请先加载模型。"
        }
    },
    "prefix": {
        "en": {
            "placeholder": "System prompt (optional)"
        },
        "zh": {
            "placeholder": "系统提示词（非必填）"
        }
    },
    "query": {
        "en": {
            "placeholder": "Input..."
        },
        "zh": {
            "placeholder": "输入..."
        }
    },
    "submit_btn": {
        "en": {
            "value": "Submit"
        },
        "zh": {
            "value": "提交"
        }
    },
    "clear_btn": {
        "en": {
            "value": "Clear history"
        },
        "zh": {
            "value": "清空历史"
        }
    },
    "max_length": {
        "en": {
            "label": "Maximum length"
        },
        "zh": {
            "label": "最大长度"
        }
    },
    "max_new_tokens": {
        "en": {
            "label": "Maximum new tokens"
        },
        "zh": {
            "label": "最大生成长度"
        }
    },
    "top_p": {
        "en": {
            "label": "Top-p"
        },
        "zh": {
            "label": "Top-p 采样值"
        }
    },
    "temperature": {
        "en": {
            "label": "Temperature"
        },
        "zh": {
            "label": "温度系数"
        }
    }
}


ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it firstly.",
        "zh": "任务已存在，请先中断训练。"
    },
    "err_exists": {
        "en": "You have loaded a model, please unload it first.",
        "zh": "模型已存在，请先卸载模型。"
    },
    "err_no_model": {
        "en": "Please select a model.",
        "zh": "请选择模型。"
    },
    "err_no_path": {
        "en": "Model not found.",
        "zh": "模型未找到。"
    },
    "err_no_dataset": {
        "en": "Please choose a dataset.",
        "zh": "请选择数据集。"
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "zh": "训练中断，正在等待线程结束……"
    },
    "info_aborted": {
        "en": "Ready.",
        "zh": "准备就绪。"
    },
    "info_finished": {
        "en": "Finished.",
        "zh": "训练完毕。"
    },
    "info_loading": {
        "en": "Loading model...",
        "zh": "加载中……"
    },
    "info_unloading": {
        "en": "Unloading model...",
        "zh": "卸载中……"
    },
    "info_loaded": {
        "en": "Model loaded, now you can chat with your model!",
        "zh": "模型已加载，可以开始聊天了！"
    },
    "info_unloaded": {
        "en": "Model unloaded.",
        "zh": "模型已卸载。"
    }
}
