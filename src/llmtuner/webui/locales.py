LOCALES = {
    "lang": {"en": {"label": "Lang"}, "ru": {"label": "Русский"}, "zh": {"label": "语言"}},
    "model_name": {"en": {"label": "Model name"}, "ru": {"label": "Название модели"}, "zh": {"label": "模型名称"}},
    "model_path": {
        "en": {"label": "Model path", "info": "Path to pretrained model or model identifier from Hugging Face."},
        "ru": {"label": "Путь к предварительно обученной модели или идентификатор модели от Hugging Face."},
        "zh": {"label": "模型路径", "info": "本地模型的文件路径或 Hugging Face 的模型标识符。"},
    },
    "finetuning_type": {
        "en": {"label": "Finetuning method"},
        "ru": {"label": "Метод дообучения"},
        "zh": {"label": "微调方法"},
    },
    "adapter_path": {"en": {"label": "Adapter path"}, "ru": {"label": "Путь к адаптеру"}, "zh": {"label": "适配器路径"}},
    "refresh_btn": {
        "en": {"value": "Refresh adapters"},
        "ru": {"value": "Обновить адаптеры"},
        "zh": {"value": "刷新适配器"},
    },
    "advanced_tab": {
        "en": {"label": "Advanced configurations"},
        "ru": {"label": "Расширенные конфигурации"},
        "zh": {"label": "高级设置"},
    },
    "quantization_bit": {
        "en": {"label": "Quantization bit", "info": "Enable 4/8-bit model quantization (QLoRA)."},
        "ru": {"label": "Уровень квантования", "info": "Включить 4/8-битное квантование модели (QLoRA)."},
        "zh": {"label": "量化等级", "info": "启用 4/8 比特模型量化（QLoRA）。"},
    },
    "template": {
        "en": {"label": "Prompt template", "info": "The template used in constructing prompts."},
        "ru": {"label": "Шаблон запроса", "info": "Шаблон, используемый при формировании запросов."},
        "zh": {"label": "提示模板", "info": "构建提示词时使用的模板"},
    },
    "rope_scaling": {
        "en": {"label": "RoPE scaling"},
        "ru": {"label": "Масштабирование RoPE"},
        "zh": {"label": "RoPE 插值方法"},
    },
    "booster": {"en": {"label": "Booster"}, "ru": {"label": "Ускоритель"}, "zh": {"label": "加速方式"}},
    "training_stage": {
        "en": {"label": "Stage", "info": "The stage to perform in training."},
        "ru": {"label": "Этап", "info": "Этап выполнения обучения."},
        "zh": {"label": "训练阶段", "info": "目前采用的训练方式。"},
    },
    "dataset_dir": {
        "en": {"label": "Data dir", "info": "Path to the data directory."},
        "ru": {"label": "Директория данных", "info": "Путь к директории данных."},
        "zh": {"label": "数据路径", "info": "数据文件夹的路径。"},
    },
    "dataset": {"en": {"label": "Dataset"}, "ru": {"label": "Набор данных"}, "zh": {"label": "数据集"}},
    "data_preview_btn": {
        "en": {"value": "Preview dataset"},
        "ru": {"value": "Просмотреть набор данных"},
        "zh": {"value": "预览数据集"},
    },
    "preview_count": {"en": {"label": "Count"}, "ru": {"label": "Количество"}, "zh": {"label": "数量"}},
    "page_index": {"en": {"label": "Page"}, "ru": {"label": "Страница"}, "zh": {"label": "页数"}},
    "prev_btn": {"en": {"value": "Prev"}, "ru": {"value": "Предыдущая"}, "zh": {"value": "上一页"}},
    "next_btn": {"en": {"value": "Next"}, "ru": {"value": "Следующая"}, "zh": {"value": "下一页"}},
    "close_btn": {"en": {"value": "Close"}, "ru": {"value": "Закрыть"}, "zh": {"value": "关闭"}},
    "preview_samples": {"en": {"label": "Samples"}, "ru": {"label": "Примеры"}, "zh": {"label": "样例"}},
    "cutoff_len": {
        "en": {"label": "Cutoff length", "info": "Max tokens in input sequence."},
        "ru": {"label": "Длина обрезки", "info": "Максимальное количество токенов во входной последовательности."},
        "zh": {"label": "截断长度", "info": "输入序列分词后的最大长度。"},
    },
    "learning_rate": {
        "en": {"label": "Learning rate", "info": "Initial learning rate for AdamW."},
        "ru": {"label": "Скорость обучения", "info": "Начальная скорость обучения для AdamW."},
        "zh": {"label": "学习率", "info": "AdamW 优化器的初始学习率。"},
    },
    "num_train_epochs": {
        "en": {"label": "Epochs", "info": "Total number of training epochs to perform."},
        "ru": {"label": "Эпохи", "info": "Общее количество эпох обучения."},
        "zh": {"label": "训练轮数", "info": "需要执行的训练总轮数。"},
    },
    "max_samples": {
        "en": {"label": "Max samples", "info": "Maximum samples per dataset."},
        "ru": {
            "label": "Максимальное количество образцов",
            "info": "Максимальное количество образцов на набор данных.",
        },
        "zh": {"label": "最大样本数", "info": "每个数据集的最大样本数。"},
    },
    "compute_type": {
        "en": {"label": "Compute type", "info": "Whether to use mixed precision training (fp16 or bf16)."},
        "ru": {"label": "Тип вычислений", "info": "Использовать ли обучение смешанной точности fp16 или bf16."},
        "zh": {"label": "计算类型", "info": "是否使用混合精度训练（fp16 或 bf16）。"},
    },
    "batch_size": {
        "en": {"label": "Batch size", "info": "Number of samples processed on each GPU."},
        "ru": {"label": "Размер пакета", "info": "Количество образцов для обработки на каждом GPU."},
        "zh": {"label": "批处理大小", "info": "每个 GPU 处理的样本数量。"},
    },
    "gradient_accumulation_steps": {
        "en": {"label": "Gradient accumulation", "info": "Number of steps for gradient accumulation."},
        "ru": {"label": "Накопление градиента", "info": "Количество шагов накопления градиента."},
        "zh": {"label": "梯度累积", "info": "梯度累积的步数。"},
    },
    "lr_scheduler_type": {
        "en": {"label": "LR scheduler", "info": "Name of the learning rate scheduler."},
        "ru": {"label": "Планировщик скорости обучения", "info": "Название планировщика скорости обучения."},
        "zh": {"label": "学习率调节器", "info": "学习率调度器的名称。"},
    },
    "max_grad_norm": {
        "en": {"label": "Maximum gradient norm", "info": "Norm for gradient clipping."},
        "ru": {"label": "Максимальная норма градиента", "info": "Норма для обрезки градиента."},
        "zh": {"label": "最大梯度范数", "info": "用于梯度裁剪的范数。"},
    },
    "val_size": {
        "en": {"label": "Val size", "info": "Proportion of data in the dev set."},
        "ru": {"label": "Размер валидации", "info": "Пропорция данных в наборе для разработки."},
        "zh": {"label": "验证集比例", "info": "验证集占全部样本的百分比。"},
    },
    "extra_tab": {
        "en": {"label": "Extra configurations"},
        "ru": {"label": "Дополнительные конфигурации"},
        "zh": {"label": "其它参数设置"},
    },
    "logging_steps": {
        "en": {"label": "Logging steps", "info": "Number of steps between two logs."},
        "ru": {"label": "Шаги логирования", "info": "Количество шагов между двумя записями в журнале."},
        "zh": {"label": "日志间隔", "info": "每两次日志输出间的更新步数。"},
    },
    "save_steps": {
        "en": {"label": "Save steps", "info": "Number of steps between two checkpoints."},
        "ru": {"label": "Шаги сохранения", "info": "Количество шагов между двумя контрольными точками."},
        "zh": {"label": "保存间隔", "info": "每两次断点保存间的更新步数。"},
    },
    "warmup_steps": {
        "en": {"label": "Warmup steps", "info": "Number of steps used for warmup."},
        "ru": {"label": "Шаги прогрева", "info": "Количество шагов, используемых для прогрева."},
        "zh": {"label": "预热步数", "info": "学习率预热采用的步数。"},
    },
    "neftune_alpha": {
        "en": {"label": "NEFTune Alpha", "info": "Magnitude of noise adding to embedding vectors."},
        "ru": {"label": "NEFTune Alpha", "info": "Величина шума, добавляемого к векторам вложений."},
        "zh": {"label": "NEFTune 噪声参数", "info": "嵌入向量所添加的噪声大小。"},
    },
    "sft_packing": {
        "en": {
            "label": "Pack sequences",
            "info": "Pack sequences into samples of fixed length in supervised fine-tuning.",
        },
        "ru": {
            "label": "Упаковка последовательностей",
            "info": "Упаковка последовательностей в образцы фиксированной длины при контролируемой тонкой настройке.",
        },
        "zh": {"label": "序列打包", "info": "在指令监督微调阶段将序列打包为相同长度的样本。"},
    },
    "upcast_layernorm": {
        "en": {"label": "Upcast LayerNorm", "info": "Upcast weights of layernorm in float32."},
        "ru": {"label": "Приведение весов LayerNorm", "info": "Приведение весов LayerNorm к float32."},
        "zh": {"label": "缩放归一化层", "info": "将归一化层权重缩放至 32 位精度。"},
    },
    "lora_tab": {
        "en": {"label": "LoRA configurations"},
        "ru": {"label": "Конфигурации LoRA"},
        "zh": {"label": "LoRA 参数设置"},
    },
    "lora_rank": {
        "en": {"label": "LoRA rank", "info": "The rank of LoRA matrices."},
        "ru": {"label": "Ранг матриц LoRA", "info": "Ранг матриц LoRA."},
        "zh": {"label": "LoRA 秩", "info": "LoRA 矩阵的秩。"},
    },
    "lora_dropout": {
        "en": {"label": "LoRA Dropout", "info": "Dropout ratio of LoRA weights."},
        "ru": {"label": "Вероятность отсева LoRA", "info": "Вероятность отсева весов LoRA."},
        "zh": {"label": "LoRA 随机丢弃", "info": "LoRA 权重随机丢弃的概率。"},
    },
    "lora_target": {
        "en": {
            "label": "LoRA modules (optional)",
            "info": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules.",
        },
        "ru": {
            "label": "Модули LoRA (опционально)",
            "info": "Имена целевых модулей для применения LoRA. Используйте запятые для разделения нескольких модулей.",
        },
        "zh": {"label": "LoRA 作用模块（非必填）", "info": "应用 LoRA 的目标模块名称。使用英文逗号分隔多个名称。"},
    },
    "additional_target": {
        "en": {
            "label": "Additional modules (optional)",
            "info": "Name(s) of modules apart from LoRA layers to be set as trainable. Use commas to separate multiple modules.",
        },
        "ru": {
            "label": "Дополнительные модули (опционально)",
            "info": "Имена модулей, кроме слоев LoRA, которые следует установить в качестве обучаемых. Используйте запятые для разделения нескольких модулей.",
        },
        "zh": {"label": "附加模块（非必填）", "info": "除 LoRA 层以外的可训练模块名称。使用英文逗号分隔多个名称。"},
    },
    "create_new_adapter": {
        "en": {
            "label": "Create new adapter",
            "info": "Whether to create a new adapter with randomly initialized weight or not.",
        },
        "ru": {
            "label": "Создать новый адаптер",
            "info": "Создать новый адаптер с случайной инициализацией веса или нет.",
        },
        "zh": {"label": "新建适配器", "info": "是否创建一个经过随机初始化的新适配器。"},
    },
    "rlhf_tab": {
        "en": {"label": "RLHF configurations"},
        "ru": {"label": "Конфигурации RLHF"},
        "zh": {"label": "RLHF 参数设置"},
    },
    "dpo_beta": {
        "en": {"label": "DPO beta", "info": "Value of the beta parameter in the DPO loss."},
        "ru": {"label": "DPO бета", "info": "Значение параметра бета в функции потерь DPO."},
        "zh": {"label": "DPO beta 参数", "info": "DPO 损失函数中 beta 超参数大小。"},
    },
    "dpo_ftx": {
        "en": {"label": "DPO-ftx weight", "info": "The weight of SFT loss in the DPO-ftx."},
        "ru": {"label": "Вес DPO-ftx", "info": "Вес функции потерь SFT в DPO-ftx."},
        "zh": {"label": "DPO-ftx 权重", "info": "DPO-ftx 中 SFT 损失的权重大小。"},
    },
    "reward_model": {
        "en": {
            "label": "Reward model",
            "info": "Adapter of the reward model for PPO training. (Needs to refresh adapters)",
        },
        "ru": {
            "label": "Модель вознаграждения",
            "info": "Адаптер модели вознаграждения для обучения PPO. (Необходимо обновить адаптеры)",
        },
        "zh": {"label": "奖励模型", "info": "PPO 训练中奖励模型的适配器路径。（需要刷新适配器）"},
    },
    "cmd_preview_btn": {
        "en": {"value": "Preview command"},
        "ru": {"value": "Просмотр команды"},
        "zh": {"value": "预览命令"},
    },
    "start_btn": {"en": {"value": "Start"}, "ru": {"value": "Начать"}, "zh": {"value": "开始"}},
    "stop_btn": {"en": {"value": "Abort"}, "ru": {"value": "Прервать"}, "zh": {"value": "中断"}},
    "output_dir": {
        "en": {"label": "Output dir", "info": "Directory for saving results."},
        "ru": {"label": "Выходной каталог", "info": "Каталог для сохранения результатов."},
        "zh": {"label": "输出目录", "info": "保存结果的路径。"},
    },
    "output_box": {"en": {"value": "Ready."}, "ru": {"value": "Готово."}, "zh": {"value": "准备就绪。"}},
    "loss_viewer": {"en": {"label": "Loss"}, "ru": {"label": "Потери"}, "zh": {"label": "损失"}},
    "predict": {
        "en": {"label": "Save predictions"},
        "ru": {"label": "Сохранить предсказания"},
        "zh": {"label": "保存预测结果"},
    },
    "load_btn": {"en": {"value": "Load model"}, "ru": {"value": "Загрузить модель"}, "zh": {"value": "加载模型"}},
    "unload_btn": {"en": {"value": "Unload model"}, "ru": {"value": "Выгрузить модель"}, "zh": {"value": "卸载模型"}},
    "info_box": {
        "en": {"value": "Model unloaded, please load a model first."},
        "ru": {"value": "Модель не загружена, загрузите модель сначала."},
        "zh": {"value": "模型未加载，请先加载模型。"},
    },
    "system": {
        "en": {"placeholder": "System prompt (optional)"},
        "ru": {"placeholder": "Системный запрос (по желанию)"},
        "zh": {"placeholder": "系统提示词（非必填）"},
    },
    "tools": {
        "en": {"placeholder": "Tools (optional)"},
        "ru": {"placeholder": "Инструменты (по желанию)"},
        "zh": {"placeholder": "工具列表（非必填）"},
    },
    "query": {"en": {"placeholder": "Input..."}, "ru": {"placeholder": "Ввод..."}, "zh": {"placeholder": "输入..."}},
    "submit_btn": {"en": {"value": "Submit"}, "ru": {"value": "Отправить"}, "zh": {"value": "提交"}},
    "clear_btn": {"en": {"value": "Clear history"}, "ru": {"value": "Очистить историю"}, "zh": {"value": "清空历史"}},
    "max_length": {"en": {"label": "Maximum length"}, "ru": {"label": "Максимальная длина"}, "zh": {"label": "最大长度"}},
    "max_new_tokens": {
        "en": {"label": "Maximum new tokens"},
        "ru": {"label": "Максимальное количество новых токенов"},
        "zh": {"label": "最大生成长度"},
    },
    "top_p": {"en": {"label": "Top-p"}, "ru": {"label": "Лучшие-p"}, "zh": {"label": "Top-p 采样值"}},
    "temperature": {"en": {"label": "Temperature"}, "ru": {"label": "Температура"}, "zh": {"label": "温度系数"}},
    "max_shard_size": {
        "en": {"label": "Max shard size (GB)", "info": "The maximum size for a model file."},
        "ru": {"label": "Максимальный размер фрагмента (ГБ)", "info": "Максимальный размер файла модели."},
        "zh": {"label": "最大分块大小（GB）", "info": "单个模型文件的最大大小。"},
    },
    "export_quantization_bit": {
        "en": {"label": "Export quantization bit.", "info": "Quantizing the exported model."},
        "ru": {"label": "Экспорт бита квантования", "info": "Квантование экспортируемой модели."},
        "zh": {"label": "导出量化等级", "info": "量化导出模型。"},
    },
    "export_quantization_dataset": {
        "en": {"label": "Export quantization dataset.", "info": "The calibration dataset used for quantization."},
        "ru": {
            "label": "Экспорт набора данных для квантования",
            "info": "Набор данных калибровки, используемый для квантования.",
        },
        "zh": {"label": "导出量化数据集", "info": "量化过程中使用的校准数据集。"},
    },
    "export_dir": {
        "en": {"label": "Export dir", "info": "Directory to save exported model."},
        "ru": {"label": "Каталог экспорта", "info": "Каталог для сохранения экспортированной модели."},
        "zh": {"label": "导出目录", "info": "保存导出模型的文件夹路径。"},
    },
    "export_btn": {"en": {"value": "Export"}, "ru": {"value": "Экспорт"}, "zh": {"value": "开始导出"}},
}


ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it first.",
        "ru": "Процесс уже запущен, пожалуйста, сначала прервите его.",
        "zh": "任务已存在，请先中断训练。",
    },
    "err_exists": {
        "en": "You have loaded a model, please unload it first.",
        "ru": "Вы загрузили модель, сначала разгрузите ее.",
        "zh": "模型已存在，请先卸载模型。",
    },
    "err_no_model": {"en": "Please select a model.", "ru": "Пожалуйста, выберите модель.", "zh": "请选择模型。"},
    "err_no_path": {"en": "Model not found.", "ru": "Модель не найдена.", "zh": "模型未找到。"},
    "err_no_dataset": {"en": "Please choose a dataset.", "ru": "Пожалуйста, выберите набор данных.", "zh": "请选择数据集。"},
    "err_no_adapter": {"en": "Please select an adapter.", "ru": "Пожалуйста, выберите адаптер.", "zh": "请选择一个适配器。"},
    "err_no_export_dir": {
        "en": "Please provide export dir.",
        "ru": "Пожалуйста, укажите каталог для экспорта.",
        "zh": "请填写导出目录",
    },
    "err_failed": {"en": "Failed.", "ru": "Ошибка.", "zh": "训练出错。"},
    "err_demo": {
        "en": "Training is unavailable in demo mode, duplicate the space to a private one first.",
        "ru": "Обучение недоступно в демонстрационном режиме, сначала скопируйте пространство в частное.",
        "zh": "展示模式不支持训练，请先复制到私人空间。",
    },
    "err_device_count": {
        "en": "Multiple GPUs are not supported yet.",
        "ru": "Пока не поддерживается множественные GPU.",
        "zh": "尚不支持多 GPU 训练。",
    },
    "err_tool_name": {"en": "Tool name not found.", "ru": "Имя инструмента не найдено.", "zh": "工具名称未找到。"},
    "err_json_schema": {"en": "Invalid JSON schema.", "ru": "Неверная схема JSON.", "zh": "Json 格式错误。"},
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "ru": "Прервано, ожидание завершения...",
        "zh": "训练中断，正在等待线程结束……",
    },
    "info_aborted": {"en": "Ready.", "ru": "Готово.", "zh": "准备就绪。"},
    "info_finished": {"en": "Finished.", "ru": "Завершено.", "zh": "训练完毕。"},
    "info_loading": {"en": "Loading model...", "ru": "Загрузка модели...", "zh": "加载中……"},
    "info_unloading": {"en": "Unloading model...", "ru": "Выгрузка модели...", "zh": "卸载中……"},
    "info_loaded": {
        "en": "Model loaded, now you can chat with your model!",
        "ru": "Модель загружена, теперь вы можете общаться с вашей моделью!",
        "zh": "模型已加载，可以开始聊天了！",
    },
    "info_unloaded": {"en": "Model unloaded.", "ru": "Модель выгружена.", "zh": "模型已卸载。"},
    "info_exporting": {"en": "Exporting model...", "ru": "Экспорт модели...", "zh": "正在导出模型……"},
    "info_exported": {"en": "Model exported.", "ru": "Модель экспортирована.", "zh": "模型导出完成。"},
}
