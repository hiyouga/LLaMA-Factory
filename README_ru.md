![# LLaMA Factory](assets/logo.png)

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Factory)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![PyPI](https://img.shields.io/pypi/v/llmtuner)](https://pypi.org/project/llmtuner/)
[![Downloads](https://static.pepy.tech/badge/llmtuner)](https://pypi.org/project/llmtuner/)
[![Citation](https://img.shields.io/badge/citation-34-green)](#projects-using-llama-factory)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
[![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
[![Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)

👋 Присоединяйтесь к нам [WeChat](assets/wechat.jpg).

\[ English | [中文](README_zh.md) \]

**Точная настройка большой языковой модели может быть настолько простой, как...**

https://github.com/hiyouga/LLaMA-Factory/assets/16256802/9840a653-7e9c-41c8-ae89-7ace5698baf6

Выберите расположение:

- **Colab**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
- **Локально**: Используйте эту инструкцию [usage](#getting-started)

## Содержание

- [Возможности](#возможности)
- [Производительность](#производительность)
- [Изменения](#изменения)
- [Поддерживаемые модели](#поддерживаемые-модели)
- [Поддерживаемые подходы к обучению](#поддерживаемые-подходы-к-обучению)
- [Предоставленные наборы данных](#предоставленные-наборы-данных)
- [Требования](#требования)
- [Начало работы](#начало-работы)
- [Проекты использующие LLaMA Factory](#проекты-использующие-llama-factory)
- [Лицензия](#лицензия)
- [Цитирование](#цитирование)
- [Благодарность](#благодарность)

## Возможности

- **Разные модели**: LLaMA, Mistral, Mixtral-MoE, Qwen, Yi, Gemma, Baichuan, ChatGLM, Phi, и другие.
- **Интегрированные подходы**: (Continuous) pre-training, supervised fine-tuning, reward modeling, PPO, DPO и ORPO.
- **Масштабируемые ресурсы**: 32-bit full-tuning, 16-bit freeze-tuning, 16-bit LoRA и 2/4/8-bit QLoRA через AQLM/AWQ/GPTQ/LLM.int8.
- **Расширенные алгоритмы**: GaLore, BAdam, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ и Agent tuning.
- **Практические трюки**: FlashAttention-2, Unsloth, RoPE scaling, NEFTune и rsLoRA.
- **Мониторинг экспериментов**: LlamaBoard, TensorBoard, Wandb, MLflow, и другие.
- **Более быстрый вывод**: OpenAI совместимый API, Gradio UI и CLI с vLLM worker.

## Производительность

В сравнении с ChatGLM's [P-Tuning](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning), LoRA tuning в LLaMA Factory обеспечивает скорость обучения в **3.7 раз быстрее** с большим баллом в Rouge на задаче генерации рекламного текста. Используя технику 4-bit quantization, LLaMA Factory's QLoRA повышает еще больше эффективность использования GPU памяти.

![benchmark](assets/benchmark.svg)

<details><summary>Определения</summary>

- **Скорость обучения**: количество обучающих выборок обрабатываемые в секунду во время обучения. (bs=4, cutoff_len=1024)
- **Балл Rouge**: Балл Rouge-2 набора разработки задачи [генерации рекламного текста](https://aclanthology.org/D19-1321.pdf). (bs=4, cutoff_len=1024)
- **Память GPU**: Максимальное использование памяти GPU в обучении 4-bit quantized. (bs=1, cutoff_len=1024)
- We adopt `pre_seq_len=128` for ChatGLM's P-Tuning and `lora_rank=32` for LLaMA Factory's LoRA tuning.

</details>

## Изменения

[24/04/21] We supported **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)** according to [AstraMindAI's implementation](https://github.com/astramind-ai/Mixture-of-depths). See `examples/extras/mod` for usage.

[24/04/19] We supported **Meta Llama 3** model series.

[24/04/16] We supported **[BAdam](https://arxiv.org/abs/2404.02827)**. See `examples/extras/badam` for usage.

[24/04/16] We supported **[unsloth](https://github.com/unslothai/unsloth)**'s long-sequence training (Llama-2-7B-56k within 24GB). It achieves **117%** speed and **50%** memory compared with FlashAttention-2, more benchmarks can be found in [this page](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison).

<details><summary>Full Changelog</summary>

[24/03/31] We supported **[ORPO](https://arxiv.org/abs/2403.07691)**. See `examples/lora_single_gpu` for usage.

[24/03/21] Our paper "[LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models](https://arxiv.org/abs/2403.13372)" is available at arXiv!

[24/03/20] We supported **FSDP+QLoRA** that fine-tunes a 70B model on 2x24GB GPUs. See `examples/extras/fsdp_qlora` for usage.

[24/03/13] We supported **[LoRA+](https://arxiv.org/abs/2402.12354)**. See `examples/extras/loraplus` for usage.

[24/03/07] We supported gradient low-rank projection (**[GaLore](https://arxiv.org/abs/2403.03507)**) algorithm. See `examples/extras/galore` for usage.

[24/03/07] We integrated **[vLLM](https://github.com/vllm-project/vllm)** for faster and concurrent inference. Try `--infer_backend vllm` to enjoy **270%** inference speed. (LoRA is not yet supported, merge it first.)

[24/02/28] We supported weight-decomposed LoRA (**[DoRA](https://arxiv.org/abs/2402.09353)**). Try `--use_dora` to activate DoRA training.

[24/02/15] We supported **block expansion** proposed by [LLaMA Pro](https://github.com/TencentARC/LLaMA-Pro). See `examples/extras/llama_pro` for usage.

[24/02/05] Qwen1.5 (Qwen2 beta version) series models are supported in LLaMA-Factory. Check this [blog post](https://qwenlm.github.io/blog/qwen1.5/) for details.

[24/01/18] We supported **agent tuning** for most models, equipping model with tool using abilities by fine-tuning with `--dataset glaive_toolcall`.

[23/12/23] We supported **[unsloth](https://github.com/unslothai/unsloth)**'s implementation to boost LoRA tuning for the LLaMA, Mistral and Yi models. Try `--use_unsloth` argument to activate unsloth patch. It achieves **170%** speed in our benchmark, check [this page](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison) for details.

[23/12/12] We supported fine-tuning the latest MoE model **[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)** in our framework. See hardware requirement [here](#hardware-requirement).

[23/12/01] We supported downloading pre-trained models and datasets from the **[ModelScope Hub](https://modelscope.cn/models)** for Chinese mainland users. See [this tutorial](#use-modelscope-hub-optional) for usage.

[23/10/21] We supported **[NEFTune](https://arxiv.org/abs/2310.05914)** trick for fine-tuning. Try `--neftune_noise_alpha` argument to activate NEFTune, e.g., `--neftune_noise_alpha 5`.

[23/09/27] We supported **$S^2$-Attn** proposed by [LongLoRA](https://github.com/dvlab-research/LongLoRA) for the LLaMA models. Try `--shift_attn` argument to enable shift short attention.

[23/09/23] We integrated MMLU, C-Eval and CMMLU benchmarks in this repo. See [this example](#evaluation) to evaluate your models.

[23/09/10] We supported **[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)**. Try `--flash_attn` argument to enable FlashAttention-2 if you are using RTX4090, A100 or H100 GPUs.

[23/08/12] We supported **RoPE scaling** to extend the context length of the LLaMA models. Try `--rope_scaling linear` argument in training and `--rope_scaling dynamic` argument at inference to extrapolate the position embeddings.

[23/08/11] We supported **[DPO training](https://arxiv.org/abs/2305.18290)** for instruction-tuned models. See [this example](#dpo-training) to train your models.

[23/07/31] We supported **dataset streaming**. Try `--streaming` and `--max_steps 10000` arguments to load your dataset in streaming mode.

[23/07/29] We released two instruction-tuned 13B models at Hugging Face. See these Hugging Face Repos ([LLaMA-2](https://huggingface.co/hiyouga/Llama-2-Chinese-13b-chat) / [Baichuan](https://huggingface.co/hiyouga/Baichuan-13B-sft)) for details.

[23/07/18] We developed an **all-in-one Web UI** for training, evaluation and inference. Try `train_web.py` to fine-tune models in your Web browser. Thank [@KanadeSiina](https://github.com/KanadeSiina) and [@codemayq](https://github.com/codemayq) for their efforts in the development.

[23/07/09] We released **[FastEdit](https://github.com/hiyouga/FastEdit)** ⚡🩹, an easy-to-use package for editing the factual knowledge of large language models efficiently. Please follow [FastEdit](https://github.com/hiyouga/FastEdit) if you are interested.

[23/06/29] We provided a **reproducible example** of training a chat model using instruction-following datasets, see [Baichuan-7B-sft](https://huggingface.co/hiyouga/Baichuan-7B-sft) for details.

[23/06/22] We aligned the [demo API](src/api_demo.py) with the [OpenAI's](https://platform.openai.com/docs/api-reference/chat) format where you can insert the fine-tuned model in **arbitrary ChatGPT-based applications**.

[23/06/03] We supported quantized training and inference (aka **[QLoRA](https://github.com/artidoro/qlora)**). Try `--quantization_bit 4/8` argument to work with quantized models.

</details>

## Поддерживаемые модели

| Модель                                                   | Размер модели               | Стандартный модуль| Шаблон    |
| -------------------------------------------------------- | --------------------------- | ----------------- | --------- |
| [Baichuan2](https://huggingface.co/baichuan-inc)         | 7B/13B                      | W_pack            | baichuan2 |
| [BLOOM](https://huggingface.co/bigscience)               | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [BLOOMZ](https://huggingface.co/bigscience)              | 560M/1.1B/1.7B/3B/7.1B/176B | query_key_value   | -         |
| [ChatGLM3](https://huggingface.co/THUDM)                 | 6B                          | query_key_value   | chatglm3  |
| [Command-R](https://huggingface.co/CohereForAI)          | 35B/104B                    | q_proj,v_proj     | cohere    |
| [DeepSeek (MoE)](https://huggingface.co/deepseek-ai)     | 7B/16B/67B                  | q_proj,v_proj     | deepseek  |
| [Falcon](https://huggingface.co/tiiuae)                  | 7B/40B/180B                 | query_key_value   | falcon    |
| [Gemma/CodeGemma](https://huggingface.co/google)         | 2B/7B                       | q_proj,v_proj     | gemma     |
| [InternLM2](https://huggingface.co/internlm)             | 7B/20B                      | wqkv              | intern2   |
| [LLaMA](https://github.com/facebookresearch/llama)       | 7B/13B/33B/65B              | q_proj,v_proj     | -         |
| [LLaMA-2](https://huggingface.co/meta-llama)             | 7B/13B/70B                  | q_proj,v_proj     | llama2    |
| [LLaMA-3](https://huggingface.co/meta-llama)             | 8B/70B                      | q_proj,v_proj     | llama3    |
| [Mistral/Mixtral](https://huggingface.co/mistralai)      | 7B/8x7B/8x22B               | q_proj,v_proj     | mistral   |
| [OLMo](https://huggingface.co/allenai)                   | 1B/7B                       | att_proj          | olmo      |
| [Phi-1.5/2](https://huggingface.co/microsoft)            | 1.3B/2.7B                   | q_proj,v_proj     | -         |
| [Qwen](https://huggingface.co/Qwen)                      | 1.8B/7B/14B/72B             | c_attn            | qwen      |
| [Qwen1.5 (Code/MoE)](https://huggingface.co/Qwen)        | 0.5B/1.8B/4B/7B/14B/32B/72B | q_proj,v_proj     | qwen      |
| [StarCoder2](https://huggingface.co/bigcode)             | 3B/7B/15B                   | q_proj,v_proj     | -         |
| [XVERSE](https://huggingface.co/xverse)                  | 7B/13B/65B                  | q_proj,v_proj     | xverse    |
| [Yi](https://huggingface.co/01-ai)                       | 6B/9B/34B                   | q_proj,v_proj     | yi        |
| [Yuan](https://huggingface.co/IEITYuan)                  | 2B/51B/102B                 | q_proj,v_proj     | yuan      |

> [!NOTE]
> **Стандартный модуль** используется для аргумента `--lora_target`, вы можете использовать `--lora_target all` чтобы использовать все доступные модули.
>
> Для "базовой" модели, аргумент `--template` может быть выбран из `default`, `alpaca`, `vicuna` и др. Но обязательно используйте **соответствующий шаблон** для "instruct/chat" моделей.
>
> Запомните, используйте **ТОТ ЖЕ САМЫЙ** шаблон для обучения и вывода.

Пожалуйста, следуйте за [constants.py](src/llmtuner/extras/constants.py) для получения полного списка поддерживаемых моделей.

Вы можете добавить свой формат чата [template.py](src/llmtuner/data/template.py).

## Поддерживаемые подходы к обучению

| Подход                 |     Full-tuning    |    Freeze-tuning   |       LoRA         |       QLoRA        |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Pre-Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Supervised Fine-Tuning | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reward Modeling        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| PPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| DPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| ORPO Training          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Предоставленные наборы данных

<details><summary>Наборы данных для пред-обучения</summary>

- [Wiki Demo (en)](data/wiki_demo.txt)
- [RefinedWeb (en)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
- [RedPajama V2 (en)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
- [Wikipedia (en)](https://huggingface.co/datasets/olm/olm-wikipedia-20221220)
- [Wikipedia (zh)](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- [Pile (en)](https://huggingface.co/datasets/EleutherAI/pile)
- [SkyPile (zh)](https://huggingface.co/datasets/Skywork/SkyPile-150B)
- [The Stack (en)](https://huggingface.co/datasets/bigcode/the-stack)
- [StarCoder (en)](https://huggingface.co/datasets/bigcode/starcoderdata)

</details>

<details><summary>Наборы данных для контроллируемеого файн-тюнинга</summary>

- [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
- [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [Alpaca GPT4 (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Self Cognition (zh)](data/self_cognition.json)
- [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [ShareGPT (zh)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Chinese-instruction-collection)
- [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
- [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
- [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
- [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- [UltraChat (en)](https://github.com/thunlp/UltraChat)
- [LIMA (en)](https://huggingface.co/datasets/GAIR/lima)
- [OpenPlatypus (en)](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
- [CodeAlpaca 20k (en)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
- [OpenOrca (en)](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [SlimOrca (en)](https://huggingface.co/datasets/Open-Orca/SlimOrca)
- [MathInstruct (en)](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- [Firefly 1.1M (zh)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
- [Wiki QA (en)](https://huggingface.co/datasets/wiki_qa)
- [Web QA (zh)](https://huggingface.co/datasets/suolyer/webqa)
- [WebNovel (zh)](https://huggingface.co/datasets/zxbsmk/webnovel_cn)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [deepctrl (en&zh)](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
- [Ad Gen (zh)](https://huggingface.co/datasets/HasturOfficial/adgen)
- [ShareGPT Hyperfiltered (en)](https://huggingface.co/datasets/totally-not-an-llm/sharegpt-hyperfiltered-3k)
- [ShareGPT4 (en&zh)](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
- [UltraChat 200k (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [AgentInstruct (en)](https://huggingface.co/datasets/THUDM/AgentInstruct)
- [LMSYS Chat 1M (en)](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- [Evol Instruct V2 (en)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)
- [Glaive Function Calling V2 (en)](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [Cosmopedia (en)](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- [Open Assistant (de)](https://huggingface.co/datasets/mayflowergmbh/oasst_de)
- [Dolly 15k (de)](https://huggingface.co/datasets/mayflowergmbh/dolly-15k_de)
- [Alpaca GPT4 (de)](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de)
- [OpenSchnabeltier (de)](https://huggingface.co/datasets/mayflowergmbh/openschnabeltier_de)
- [Evol Instruct (de)](https://huggingface.co/datasets/mayflowergmbh/evol-instruct_de)
- [Dolphin (de)](https://huggingface.co/datasets/mayflowergmbh/dolphin_de)
- [Booksum (de)](https://huggingface.co/datasets/mayflowergmbh/booksum_de)
- [Airoboros (de)](https://huggingface.co/datasets/mayflowergmbh/airoboros-3.0_de)
- [Ultrachat (de)](https://huggingface.co/datasets/mayflowergmbh/ultra-chat_de)

</details>

<details><summary>Наборы данных на основе предпочтений</summary>

- [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Open Assistant (multilingual)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [GPT-4 Generated Data (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Orca DPO (en)](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [DPO mix (en&zh)](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)
- [Orca DPO (de)](https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de)

</details>

Некоторые наборы данных требуют подтверждения перед его использованием, мы рекомендуем предварительно авторизоваться в ваш аккаунт Hugging Face используя данные команды.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## Требования

| Обязательно  | Минимум | Рекоменд. |
| ------------ | ------- | --------- |
| python       | 3.8     | 3.10      |
| torch        | 1.13.1  | 2.2.0     |
| transformers | 4.37.2  | 4.39.3    |
| datasets     | 2.14.3  | 2.18.0    |
| accelerate   | 0.27.2  | 0.28.0    |
| peft         | 0.9.0   | 0.10.0    |
| trl          | 0.8.1   | 0.8.1     |

| Опционально  | Минимум | Рекоменд. |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.14.0    |
| bitsandbytes | 0.39.0  | 0.43.0    |
| flash-attn   | 2.3.0   | 2.5.6     |

### Требования к оборудованию

\* *estimated*

| Method            | Bits |   7B  |  13B  |  30B  |   70B  |  8x7B |  8x22B |
| ----------------- | ---- | ----- | ----- | ----- | ------ | ----- | ------ |
| Full              | AMP  | 120GB | 240GB | 600GB | 1200GB | 900GB | 2400GB |
| Full              |  16  |  60GB | 120GB | 300GB |  600GB | 400GB | 1200GB |
| Freeze            |  16  |  20GB |  40GB |  80GB |  200GB | 160GB |  400GB |
| LoRA/GaLore/BAdam |  16  |  16GB |  32GB |  64GB |  160GB | 120GB |  320GB |
| QLoRA             |   8  |  10GB |  20GB |  40GB |   80GB |  60GB |  160GB |
| QLoRA             |   4  |   6GB |  12GB |  24GB |   48GB |  30GB |   96GB |
| QLoRA             |   2  |   4GB |   8GB |  16GB |   24GB |  18GB |   48GB |

## Начало работы

### Подготовка данных

Откройте [data/README.md](data/README.md) для детальной проверки формата файлов набора данных. Вы можете использовать наборы данных HuggingFace / ModelScope hub или загрузить наборы данных на локальный диск.

> [!NOTE]
> Пожалуйста обновите файл `data/dataset_info.json` для использования собственного набора данных.

### Установка зависимостей

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
conda create -n llama_factory python=3.10
conda activate llama_factory
cd LLaMA-Factory
pip install -e .[metrics]
```

Доступные дополнительные зависимости: deepspeed, metrics, unsloth, galore, badam, vllm, bitsandbytes, gptq, awq, aqlm, qwen, modelscope, quality

<details><summary>Для пользователей Windows</summary>

If you want to enable the quantized LoRA (QLoRA) on the Windows platform, you will be required to install a pre-built version of `bitsandbytes` library, which supports CUDA 11.1 to 12.2, please select the appropriate [release version](https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels) based on your CUDA version.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

Для включения FlashAttention-2 на платформе Windows, вам нужно будет установить компилированную версию библиотеки `flash-attn`, с поддержкой CUDA 12.1 до 12.2. Пожалуйста, загрузите соответствующую версию с [flash-attention](https://github.com/bdashore3/flash-attention/releases) на основе ваших требований.

</details>

### Тренировка с LLaMA Board GUI

> [!IMPORTANT]
> LLaMA Board GUI поддерживает обучение только на одном GPU, используйте [CLI](#command-line-interface) для распределенного обучения.

#### Используя локальное окружение

```bash
export CUDA_VISIBLE_DEVICES=0 # `set CUDA_VISIBLE_DEVICES=0` для Windows
export GRADIO_SERVER_PORT=7860 # `set GRADIO_SERVER_PORT=7860` для Windows
python src/train_web.py # или python -m llmtuner.webui.interface
```

<details><summary>Для пользователей Alibaba Cloud</summary>

If you encountered display problems in LLaMA Board on Alibaba Cloud, try using the following command to set environment variables before starting LLaMA Board:

```bash
export GRADIO_ROOT_PATH=/${JUPYTER_NAME}/proxy/7860/
```

</details>

#### Используя Docker

```bash
docker build -f ./Dockerfile -t llama-factory:latest .
docker run --gpus=all \
    -v ./hf_cache:/root/.cache/huggingface/ \
    -v ./data:/app/data \
    -v ./output:/app/output \
    -e CUDA_VISIBLE_DEVICES=0 \
    -p 7860:7860 \
    --shm-size 16G \
    --name llama_factory \
    -d llama-factory:latest
```

#### Используя Docker Compose

```bash
docker compose -f ./docker-compose.yml up -d
```

<details><summary>Details about volume</summary>

- hf_cache: Utilize Hugging Face cache on the host machine. Reassignable if a cache already exists in a different directory.
- data: Place datasets on this dir of the host machine so that they can be selected on LLaMA Board GUI.
- output: Set export dir to this location so that the merged result can be accessed directly on the host machine.

</details>

### Тренировка через командную строку

See [examples/README.md](examples/README.md) for usage.

Use `python src/train_bash.py -h` to display arguments description.

### Публикация в OpenAI совместимом API и vLLM

```bash
CUDA_VISIBLE_DEVICES=0,1 API_PORT=8000 python src/api_demo.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --template llama3 \
    --infer_backend vllm \
    --vllm_enforce_eager
```

### Загрузка с ModelScope Hub

Если у вас есть проблемы с загрузкой моделей с Hugging Face, вы можете использовать ModelScope.

```bash
export USE_MODELSCOPE_HUB=1 # `set USE_MODELSCOPE_HUB=1` для Windows
```

Train the model by specifying a model ID of the ModelScope Hub as the `--model_name_or_path`. You can find a full list of model IDs at [ModelScope Hub](https://modelscope.cn/models), e.g., `LLM-Research/Meta-Llama-3-8B-Instruct`.

## Проекты использующие LLaMA Factory

Если у вас есть проекты, которые нужно включить здесь, сообщите нам по email или создайте pull request.

<details><summary>Нажмите для просмотра</summary>

1. Wang et al. ESRL: Efficient Sampling-based Reinforcement Learning for Sequence Generation. 2023. [[arxiv]](https://arxiv.org/abs/2308.02223)
1. Yu et al. Open, Closed, or Small Language Models for Text Classification? 2023. [[arxiv]](https://arxiv.org/abs/2308.10092)
1. Wang et al. UbiPhysio: Support Daily Functioning, Fitness, and Rehabilitation with Action Understanding and Feedback in Natural Language. 2023. [[arxiv]](https://arxiv.org/abs/2308.10526)
1. Luceri et al. Leveraging Large Language Models to Detect Influence Campaigns in Social Media. 2023. [[arxiv]](https://arxiv.org/abs/2311.07816)
1. Zhang et al. Alleviating Hallucinations of Large Language Models through Induced Hallucinations. 2023. [[arxiv]](https://arxiv.org/abs/2312.15710)
1. Wang et al. Know Your Needs Better: Towards Structured Understanding of Marketer Demands with Analogical Reasoning Augmented LLMs. 2024. [[arxiv]](https://arxiv.org/abs/2401.04319)
1. Wang et al. CANDLE: Iterative Conceptualization and Instantiation Distillation from Large Language Models for Commonsense Reasoning. 2024. [[arxiv]](https://arxiv.org/abs/2401.07286)
1. Choi et al. FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs. 2024. [[arxiv]](https://arxiv.org/abs/2402.05904)
1. Zhang et al. AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts. 2024. [[arxiv]](https://arxiv.org/abs/2402.07625)
1. Lyu et al. KnowTuning: Knowledge-aware Fine-tuning for Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.11176)
1. Yang et al. LaCo: Large Language Model Pruning via Layer Collaps. 2024. [[arxiv]](https://arxiv.org/abs/2402.11187)
1. Bhardwaj et al. Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task Arithmetic. 2024. [[arxiv]](https://arxiv.org/abs/2402.11746)
1. Yang et al. Enhancing Empathetic Response Generation by Augmenting LLMs with Small-scale Empathetic Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.11801)
1. Yi et al. Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding. 2024. [[arxiv]](https://arxiv.org/abs/2402.11809)
1. Cao et al. Head-wise Shareable Attention for Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.11819)
1. Zhang et al. Enhancing Multilingual Capabilities of Large Language Models through Self-Distillation from Resource-Rich Languages. 2024. [[arxiv]](https://arxiv.org/abs/2402.12204)
1. Kim et al. Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.14714)
1. Yu et al. KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.15043)
1. Huang et al. Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning. 2024. [[arxiv]](https://arxiv.org/abs/2403.02333)
1. Duan et al. Negating Negatives: Alignment without Human Positive Samples via Distributional Dispreference Optimization. 2024. [[arxiv]](https://arxiv.org/abs/2403.03419)
1. Xie and Schwertfeger. Empowering Robotics with Large Language Models: osmAG Map Comprehension with LLMs. 2024. [[arxiv]](https://arxiv.org/abs/2403.08228)
1. Zhang et al. EDT: Improving Large Language Models' Generation by Entropy-based Dynamic Temperature Sampling. 2024. [[arxiv]](https://arxiv.org/abs/2403.14541)
1. Weller et al. FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions. 2024. [[arxiv]](https://arxiv.org/abs/2403.15246)
1. Hongbin Na. CBT-LLM: A Chinese Large Language Model for Cognitive Behavioral Therapy-based Mental Health Question Answering. 2024. [[arxiv]](https://arxiv.org/abs/2403.16008)
1. Zan et al. CodeS: Natural Language to Code Repository via Multi-Layer Sketch. 2024. [[arxiv]](https://arxiv.org/abs/2403.16443)
1. Liu et al. Extensive Self-Contrast Enables Feedback-Free Language Model Alignment. 2024. [[arxiv]](https://arxiv.org/abs/2404.00604)
1. Luo et al. BAdam: A Memory Efficient Full Parameter Training Method for Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2404.02827)
1. Du et al. Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model. 2024. [[arxiv]](https://arxiv.org/abs/2404.04167)
1. Liu et al. Dynamic Generation of Personalities with Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2404.07084)
1. **[StarWhisper](https://github.com/Yu-Yang-Li/StarWhisper)**: A large language model for Astronomy, based on ChatGLM2-6B and Qwen-14B.
1. **[DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)**: A large language model specialized in Chinese legal domain, based on Baichuan-13B, is capable of retrieving and reasoning on legal knowledge.
1. **[Sunsimiao](https://github.com/thomas-yanxin/Sunsimiao)**: A large language model specialized in Chinese medical domain, based on Baichuan-7B and ChatGLM-6B.
1. **[CareGPT](https://github.com/WangRongsheng/CareGPT)**: A series of large language models for Chinese medical domain, based on LLaMA2-7B and Baichuan-13B.
1. **[MachineMindset](https://github.com/PKU-YuanGroup/Machine-Mindset/)**: A series of MBTI Personality large language models, capable of giving any LLM 16 different personality types based on different datasets and training methods.

</details>

## Лицензия

Данный репозиторий лицензирован как [Apache-2.0 License](LICENSE).

Пожалуйста, соблюдайте лицензии моделей на веса: [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/Community%20License%20for%20Baichuan%202%20Model.pdf) / [BLOOM](https://huggingface.co/spaces/bigscience/license) / [ChatGLM3](https://github.com/THUDM/ChatGLM3/blob/main/MODEL_LICENSE) / [Command-R](https://cohere.com/c4ai-cc-by-nc-license) / [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/LICENSE-MODEL) / [Falcon](https://huggingface.co/tiiuae/falcon-180B/blob/main/LICENSE.txt) / [Gemma](https://ai.google.dev/gemma/terms) / [InternLM2](https://github.com/InternLM/InternLM#license) / [LLaMA](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) / [LLaMA-2](https://ai.meta.com/llama/license/) / [LLaMA-3](https://llama.meta.com/llama3/license/) / [Mistral](LICENSE) / [OLMo](LICENSE) / [Phi-1.5/2](https://huggingface.co/microsoft/phi-1_5/resolve/main/Research%20License.docx) / [Qwen](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) / [StarCoder2](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) / [XVERSE](https://github.com/xverse-ai/XVERSE-13B/blob/main/MODEL_LICENSE.pdf) / [Yi](https://huggingface.co/01-ai/Yi-6B/blob/main/LICENSE) / [Yuan](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/LICENSE-Yuan)

## Цитирование

Если данная работа была полезна для вас, пожалуйста цитируйте как:

```bibtex
@article{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Yongqiang Ma},
  journal={arXiv preprint arXiv:2403.13372},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```

## Благодарность

Данный репозиторий использует [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat).Спасибо им за отличную работу.

## История звезд

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/LLaMA-Factory&type=Date)
