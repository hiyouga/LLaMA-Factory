![# LLaMA Factory](assets/logo.png)

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/hiyouga/LLaMA-Factory)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
[![Citation](https://img.shields.io/badge/citation-72-green)](#llama-factory-を使用したプロジェクト)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)
[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
[![Open in DSW](https://gallery.pai-ml.com/assets/open-in-dsw.svg)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
[![Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
[![Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)

[![GitHub Tread](https://trendshift.io/api/badge/repositories/4535)](https://trendshift.io/repositories/4535)

👋 [WeChat](assets/wechat.jpg) または [NPUユーザーグループ](assets/wechat_npu.jpg) に参加してください。

\[ [English](README.md) | [中文](README_zh.md) | 日本語 \]

**大規模言語モデルのファインチューニングは簡単です...**

https://github.com/hiyouga/LLaMA-Factory/assets/16256802/9840a653-7e9c-41c8-ae89-7ace5698baf6

選択肢：

- **Colab**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
- **PAI-DSW**: https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory
- **ローカルマシン**: [使用方法](#使用方法) を参照してください

## 目次

- [特徴](#特徴)
- [ベンチマーク](#ベンチマーク)
- [変更履歴](#変更履歴)
- [サポートされているモデル](#サポートされているモデル)
- [サポートされているトレーニングアプローチ](#サポートされているトレーニングアプローチ)
- [提供されているデータセット](#提供されているデータセット)
- [要件](#要件)
- [使用方法](#使用方法)
- [LLaMA Factoryを使用したプロジェクト](#llama-factory-を使用したプロジェクト)
- [ライセンス](#ライセンス)
- [引用](#引用)
- [謝辞](#謝辞)

## 特徴

- **さまざまなモデル**: LLaMA、LLaVA、Mistral、Mixtral-MoE、Qwen、Yi、Gemma、Baichuan、ChatGLM、Phiなど。
- **統合された方法**: （連続）事前トレーニング、（マルチモーダル）教師ありファインチューニング、報酬モデリング、PPO、DPO、KTO、ORPOなど。
- **スケーラブルなリソース**: 16ビットのフルチューニング、フリーズチューニング、LoRA、およびAQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQによる2/3/4/5/6/8ビットのQLoRA。
- **高度なアルゴリズム**: GaLore、BAdam、DoRA、LongLoRA、LLaMA Pro、Mixture-of-Depths、LoRA+、LoftQ、PiSSA、エージェントチューニング。
- **実用的なトリック**: FlashAttention-2、Unsloth、RoPEスケーリング、NEFTune、rsLoRA。
- **実験モニター**: LlamaBoard、TensorBoard、Wandb、MLflowなど。
- **高速な推論**: OpenAIスタイルのAPI、Gradio UI、CLIとvLLMワーカー。

## ベンチマーク

ChatGLMの[P-Tuning](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)と比較して、LLaMA FactoryのLoRAチューニングは、広告テキスト生成タスクで**3.7倍の高速化**を提供し、より高いRougeスコアを達成します。4ビット量子化技術を活用することで、LLaMA FactoryのQLoRAはGPUメモリの効率をさらに向上させます。

![benchmark](assets/benchmark.svg)

<details><summary>定義</summary>

- **トレーニング速度**: トレーニング中に1秒あたりに処理されるサンプル数。（bs=4、cutoff_len=1024）
- **Rougeスコア**: [広告テキスト生成](https://aclanthology.org/D19-1321.pdf)タスクの開発セットでのRouge-2スコア。（bs=4、cutoff_len=1024）
- **GPUメモリ**: 4ビット量子化トレーニングでのピークGPUメモリ使用量。（bs=1、cutoff_len=1024）
- ChatGLMのP-Tuningには`pre_seq_len=128`を、LLaMA FactoryのLoRAチューニングには`lora_rank=32`を採用しています。

</details>

## 変更履歴

[24/06/16] **[PiSSA](https://arxiv.org/abs/2404.02948)**アルゴリズムをサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/06/07] **[Qwen2](https://qwenlm.github.io/blog/qwen2/)**および**[GLM-4](https://github.com/THUDM/GLM-4)**モデルのファインチューニングをサポートしました。

[24/05/26] **[SimPO](https://arxiv.org/abs/2405.14734)**アルゴリズムをサポートしました。使用方法については[examples](examples/README.md)を参照してください。

<details><summary>完全な変更履歴</summary>

[24/05/20] **PaliGemma**シリーズモデルのファインチューニングをサポートしました。PaliGemmaモデルは事前トレーニングされたモデルであり、チャット補完のために`gemma`テンプレートでファインチューニングする必要があります。

[24/05/18] **[KTO](https://arxiv.org/abs/2402.01306)**アルゴリズムをサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/05/14] 昇騰NPUデバイスでのトレーニングと推論をサポートしました。詳細は[インストール](#インストール)セクションを参照してください。

[24/04/26] **LLaVA-1.5**マルチモーダルLLMのファインチューニングをサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/04/22] 無料のT4 GPUでLlama-3モデルをファインチューニングするための**[Colabノートブック](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)**を提供しました。LLaMA Factoryを使用してファインチューニングされた2つのLlama-3派生モデルがHugging Faceで利用可能です。詳細は[Llama3-8B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat)および[Llama3-Chinese](https://huggingface.co/zhichen/Llama3-Chinese)を参照してください。

[24/04/21] **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)**をサポートしました。[AstraMindAIの実装](https://github.com/astramind-ai/Mixture-of-depths)に基づいています。使用方法については[examples](examples/README.md)を参照してください。

[24/04/16] **[BAdam](https://arxiv.org/abs/2404.02827)**をサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/04/16] **[unsloth](https://github.com/unslothai/unsloth)**の長シーケンストレーニング（Llama-2-7B-56kを24GBで実行）をサポートしました。FlashAttention-2と比較して**117%**の速度と**50%**のメモリを達成します。詳細なベンチマークは[このページ](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison)で確認できます。

[24/03/31] **[ORPO](https://arxiv.org/abs/2403.07691)**をサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/03/21] 私たちの論文"[LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models](https://arxiv.org/abs/2403.13372)"がarXivで公開されました！

[24/03/20] **FSDP+QLoRA**をサポートしました。70Bモデルを2x24GB GPUでファインチューニングできます。使用方法については[examples](examples/README.md)を参照してください。

[24/03/13] **[LoRA+](https://arxiv.org/abs/2402.12354)**をサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/03/07] 勾配低ランク投影（**[GaLore](https://arxiv.org/abs/2403.03507)**）アルゴリズムをサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/03/07] **[vLLM](https://github.com/vllm-project/vllm)**を統合し、より高速で同時実行可能な推論を実現しました。`infer_backend: vllm`を試して**270%**の推論速度を体験してください。

[24/02/28] 重み分解LoRA（**[DoRA](https://arxiv.org/abs/2402.09353)**）をサポートしました。`use_dora: true`を試してDoRAトレーニングを有効にしてください。

[24/02/15] **LLaMA Pro**が提案する**ブロック拡張**をサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[24/02/05] Qwen1.5（Qwen2ベータ版）シリーズモデルがLLaMA-Factoryでサポートされました。詳細は[このブログ投稿](https://qwenlm.github.io/blog/qwen1.5/)を参照してください。

[24/01/18] ほとんどのモデルに対して**エージェントチューニング**をサポートしました。`dataset: glaive_toolcall_en`でファインチューニングすることで、ツール使用能力をモデルに付与できます。

[23/12/23] **[unsloth](https://github.com/unslothai/unsloth)**の実装をサポートし、LLaMA、Mistral、YiモデルのLoRAチューニングを高速化しました。`use_unsloth: true`引数を試してunslothパッチを有効にしてください。ベンチマークでは**170%**の速度を達成しています。詳細は[このページ](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison)で確認できます。

[23/12/12] 最新のMoEモデル**[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)**のファインチューニングをサポートしました。ハードウェア要件については[こちら](#ハードウェア要件)を参照してください。

[23/12/01] **[ModelScope Hub](https://modelscope.cn/models)**から事前トレーニングされたモデルとデータセットをダウンロードすることをサポートしました。使用方法については[このチュートリアル](#modelscope-hubからのダウンロード)を参照してください。

[23/10/21] **[NEFTune](https://arxiv.org/abs/2310.05914)**トリックをサポートしました。`neftune_noise_alpha: 5`引数を試してNEFTuneを有効にしてください。

[23/09/27] LLaMAモデルに対して[LongLoRA](https://github.com/dvlab-research/LongLoRA)が提案する**$S^2$-Attn**をサポートしました。`shift_attn: true`引数を試してシフトショートアテンションを有効にしてください。

[23/09/23] このリポジトリにMMLU、C-Eval、CMMLUベンチマークを統合しました。使用方法については[examples](examples/README.md)を参照してください。

[23/09/10] **[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)**をサポートしました。RTX4090、A100、H100 GPUを使用している場合は、`flash_attn: fa2`引数を試してFlashAttention-2を有効にしてください。

[23/08/12] LLaMAモデルのコンテキスト長を拡張するための**RoPEスケーリング**をサポートしました。トレーニング時に`rope_scaling: linear`引数を、推論時に`rope_scaling: dynamic`引数を試して位置エンベディングを外挿してください。

[23/08/11] **[DPOトレーニング](https://arxiv.org/abs/2305.18290)**をサポートしました。使用方法については[examples](examples/README.md)を参照してください。

[23/07/31] **データセットストリーミング**をサポートしました。`streaming: true`および`max_steps: 10000`引数を試してデータセットをストリーミングモードでロードしてください。

[23/07/29] Hugging Faceで2つの13B指示チューニングモデルをリリースしました。詳細はこれらのHugging Faceリポジトリ（[LLaMA-2](https://huggingface.co/hiyouga/Llama-2-Chinese-13b-chat) / [Baichuan](https://huggingface.co/hiyouga/Baichuan-13B-sft)）を参照してください。

[23/07/18] トレーニング、評価、推論のための**オールインワンWeb UI**を開発しました。`train_web.py`を試してWebブラウザでモデルをファインチューニングしてください。開発において[@KanadeSiina](https://github.com/KanadeSiina)および[@codemayq](https://github.com/codemayq)の努力に感謝します。

[23/07/09] **[FastEdit](https://github.com/hiyouga/FastEdit)** ⚡🩹をリリースしました。大規模言語モデルの事実知識を効率的に編集するための使いやすいパッケージです。興味がある場合は[FastEdit](https://github.com/hiyouga/FastEdit)をフォローしてください。

[23/06/29] 指示に従ったデータセットを使用してチャットモデルをトレーニングするための**再現可能な例**を提供しました。詳細は[Baichuan-7B-sft](https://huggingface.co/hiyouga/Baichuan-7B-sft)を参照してください。

[23/06/22] [デモAPI](src/api_demo.py)を[OpenAIの](https://platform.openai.com/docs/api-reference/chat)形式に合わせました。これにより、**任意のChatGPTベースのアプリケーション**にファインチューニングされたモデルを挿入できます。

[23/06/03] 量子化トレーニングと推論（別名**[QLoRA](https://github.com/artidoro/qlora)**）をサポートしました。使用方法については[examples](examples/README.md)を参照してください。

</details>

## サポートされているモデル

| モデル                                                        | モデルサイズ                       | テンプレート  |
| ------------------------------------------------------------ | -------------------------------- | --------- |
| [Baichuan 2](https://huggingface.co/baichuan-inc)            | 7B/13B                           | baichuan2 |
| [BLOOM/BLOOMZ](https://huggingface.co/bigscience)            | 560M/1.1B/1.7B/3B/7.1B/176B      | -         |
| [ChatGLM3](https://huggingface.co/THUDM)                     | 6B                               | chatglm3  |
| [Command R](https://huggingface.co/CohereForAI)              | 35B/104B                         | cohere    |
| [DeepSeek (Code/MoE)](https://huggingface.co/deepseek-ai)    | 7B/16B/67B/236B                  | deepseek  |
| [Falcon](https://huggingface.co/tiiuae)                      | 7B/11B/40B/180B                  | falcon    |
| [Gemma/Gemma 2/CodeGemma](https://huggingface.co/google)     | 2B/7B/9B/27B                     | gemma     |
| [GLM-4](https://huggingface.co/THUDM)                        | 9B                               | glm4      |
| [InternLM2](https://huggingface.co/internlm)                 | 7B/20B                           | intern2   |
| [Llama](https://github.com/facebookresearch/llama)           | 7B/13B/33B/65B                   | -         |
| [Llama 2](https://huggingface.co/meta-llama)                 | 7B/13B/70B                       | llama2    |
| [Llama 3/Llama 3.1](https://huggingface.co/meta-llama)       | 8B/70B                           | llama3    |
| [LLaVA-1.5](https://huggingface.co/llava-hf)                 | 7B/13B                           | vicuna    |
| [Mistral/Mixtral](https://huggingface.co/mistralai)          | 7B/8x7B/8x22B                    | mistral   |
| [OLMo](https://huggingface.co/allenai)                       | 1B/7B                            | -         |
| [PaliGemma](https://huggingface.co/google)                   | 3B                               | gemma     |
| [Phi-1.5/Phi-2](https://huggingface.co/microsoft)            | 1.3B/2.7B                        | -         |
| [Phi-3](https://huggingface.co/microsoft)                    | 4B/7B/14B                        | phi       |
| [Qwen/Qwen1.5/Qwen2 (Code/MoE)](https://huggingface.co/Qwen) | 0.5B/1.5B/4B/7B/14B/32B/72B/110B | qwen      |
| [StarCoder 2](https://huggingface.co/bigcode)                | 3B/7B/15B                        | -         |
| [XVERSE](https://huggingface.co/xverse)                      | 7B/13B/65B                       | xverse    |
| [Yi/Yi-1.5](https://huggingface.co/01-ai)                    | 6B/9B/34B                        | yi        |
| [Yi-VL](https://huggingface.co/01-ai)                        | 6B/34B                           | yi_vl     |
| [Yuan 2](https://huggingface.co/IEITYuan)                    | 2B/51B/102B                      | yuan      |

> [!NOTE]
> "ベース"モデルの場合、`template`引数は`default`、`alpaca`、`vicuna`などから選択できます。ただし、"指示/チャット"モデルの場合は**対応するテンプレート**を使用してください。
>
> トレーニングと推論で**同じ**テンプレートを使用することを忘れないでください。

サポートされているモデルの完全なリストについては、[constants.py](src/llamafactory/extras/constants.py)を参照してください。

カスタムチャットテンプレートを[template.py](src/llamafactory/data/template.py)に追加することもできます。

## サポートされているトレーニングアプローチ

| アプローチ               |     フルチューニング    |    フリーズチューニング   |       LoRA         |       QLoRA        |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| 事前トレーニング           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 教師ありファインチューニング | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 報酬モデリング        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| PPOトレーニング           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| DPOトレーニング           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| KTOトレーニング           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| ORPOトレーニング          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| SimPOトレーニング         | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## 提供されているデータセット

<details><summary>事前トレーニングデータセット</summary>

- [Wiki Demo (en)](data/wiki_demo.txt)
- [RefinedWeb (en)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
- [RedPajama V2 (en)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
- [Wikipedia (en)](https://huggingface.co/datasets/olm/olm-wikipedia-20221220)
- [Wikipedia (zh)](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
- [Pile (en)](https://huggingface.co/datasets/EleutherAI/pile)
- [SkyPile (zh)](https://huggingface.co/datasets/Skywork/SkyPile-150B)
- [FineWeb (en)](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FineWeb-Edu (en)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [The Stack (en)](https://huggingface.co/datasets/bigcode/the-stack)
- [StarCoder (en)](https://huggingface.co/datasets/bigcode/starcoderdata)

</details>

<details><summary>教師ありファインチューニングデータセット</summary>

- [Identity (en&zh)](data/identity.json)
- [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
- [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)
- [Alpaca GPT4 (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [Glaive Function Calling V2 (en&zh)](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [LIMA (en)](https://huggingface.co/datasets/GAIR/lima)
- [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
- [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
- [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
- [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
- [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
- [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
- [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
- [UltraChat (en)](https://github.com/thunlp/UltraChat)
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
- [Advertise Generating (zh)](https://huggingface.co/datasets/HasturOfficial/adgen)
- [ShareGPT Hyperfiltered (en)](https://huggingface.co/datasets/totally-not-an-llm/sharegpt-hyperfiltered-3k)
- [ShareGPT4 (en&zh)](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
- [UltraChat 200k (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [AgentInstruct (en)](https://huggingface.co/datasets/THUDM/AgentInstruct)
- [LMSYS Chat 1M (en)](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- [Evol Instruct V2 (en)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)
- [Cosmopedia (en)](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
- [STEM (zh)](https://huggingface.co/datasets/hfl/stem_zh_instruction)
- [Ruozhiba (zh)](https://huggingface.co/datasets/hfl/ruozhiba_gpt4_turbo)
- [Neo-sft (zh)](https://huggingface.co/datasets/m-a-p/neo_sft_phase2)
- [WebInstructSub (en)](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub)
- [Magpie-Pro-300K-Filtered (en)](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered)
- [LLaVA mixed (en&zh)](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k)
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

<details><summary>優先データセット</summary>

- [DPO mixed (en&zh)](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)
- [UltraFeedback (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [Orca DPO Pairs (en)](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
- [Orca DPO (de)](https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de)
- [KTO mixed (en)](https://huggingface.co/datasets/argilla/kto-mix-15k)

</details>

一部のデータセットは使用前に確認が必要です。そのため、Hugging Faceアカウントにログインすることをお勧めします。

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 要件

| 必須    | 最小 | 推奨 |
| ------------ | ------- | --------- |
| python       | 3.8     | 3.11      |
| torch        | 1.13.1  | 2.3.0     |
| transformers | 4.41.2  | 4.41.2    |
| datasets     | 2.16.0  | 2.19.2    |
| accelerate   | 0.30.1  | 0.30.1    |
| peft         | 0.11.1  | 0.11.1    |
| trl          | 0.8.6   | 0.9.4     |

| オプション     | 最小 | 推奨 |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.14.0    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.4.3     |
| flash-attn   | 2.3.0   | 2.5.9     |

### ハードウェア要件

\* *推定値*

| 方法            | ビット |   7B  |  13B  |  30B  |   70B  |  110B  |  8x7B |  8x22B |
| ----------------- | ---- | ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| フル              | AMP  | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| フル              |  16  |  60GB | 120GB | 300GB |  600GB |  900GB | 400GB | 1200GB |
| フリーズ            |  16  |  20GB |  40GB |  80GB |  200GB |  360GB | 160GB |  400GB |
| LoRA/GaLore/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  240GB | 120GB |  320GB |
| QLoRA             |   8  |  10GB |  20GB |  40GB |   80GB |  140GB |  60GB |  160GB |
| QLoRA             |   4  |   6GB |  12GB |  24GB |   48GB |   72GB |  30GB |   96GB |
| QLoRA             |   2  |   4GB |   8GB |  16GB |   24GB |   48GB |  18GB |   48GB |

## 使用方法

### インストール

> [!IMPORTANT]
> インストールは必須です。

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

利用可能な追加依存関係: torch、torch-npu、metrics、deepspeed、bitsandbytes、hqq、eetq、gptq、awq、aqlm、vllm、galore、badam、qwen、modelscope、quality

> [!TIP]
> パッケージの競合を解決するには、`pip install --no-deps -e .`を使用してください。

<details><summary>Windowsユーザー向け</summary>

Windowsプラットフォームで量子化LoRA（QLoRA）を有効にする場合、事前にビルドされた`bitsandbytes`ライブラリをインストールする必要があります。CUDA 11.1から12.2までサポートしています。CUDAバージョンに応じた適切な[リリースバージョン](https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels)を選択してください。

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

WindowsプラットフォームでFlashAttention-2を有効にするには、事前にビルドされた`flash-attn`ライブラリをインストールする必要があります。CUDA 12.1から12.2までサポートしています。必要に応じて[flash-attention](https://github.com/bdashore3/flash-attention/releases)から対応するバージョンをダウンロードしてください。

</details>

<details><summary>昇騰NPUユーザー向け</summary>

昇騰NPUデバイスでLLaMA Factoryをインストールするには、追加の依存関係を指定してください：`pip install -e ".[torch-npu,metrics]"`。さらに、**[Ascend CANN Toolkit and Kernels](https://www.hiascend.com/developer/download/community/result?module=cann)**をインストールする必要があります。[インストールチュートリアル](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/softwareinstall/instg/atlasdeploy_03_0031.html)に従うか、以下のコマンドを使用してください：

```bash
# URLをCANNバージョンとデバイスに応じて置き換えてください
# CANN Toolkitをインストール
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-toolkit_8.0.RC1.alpha001_linux-"$(uname -i)".run
bash Ascend-cann-toolkit_8.0.RC1

# CANN Kernelsをインストール
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-kernels-910b_8.0.RC1.alpha001_linux.run
bash Ascend-cann-kernels-910b_8.0.RC1.alpha001_linux.run --install

# env変数をセット
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

| 必須  | 最小 | 推奨   |
| ------------ | ------- | ----------- |
| CANN         | 8.0.RC1 | 8.0.RC1     |
| torch        | 2.1.0   | 2.1.0       |
| torch-npu    | 2.1.0   | 2.1.0.post3 |
| deepspeed    | 0.13.2  | 0.13.2      |

使用するデバイスの指定には `CUDA_VISIBLE_DEVICES` ではなく `ASCEND_RT_VISIBLE_DEVICES` を使用することを忘れないでください。

NPUデバイスでモデルを推測できない場合は、コンフィギュレーションで`do_sample: false`を設定してみてください。

ビルド済みDockerイメージのダウンロード: [32GB](http://mirrors.cn-central-221.ovaijisuan.com/detail/130.html) | [64GB](http://mirrors.cn-central-221.ovaijisuan.com/detail/131.html)

</details>

### データの準備

データセットファイルのフォーマットについては、[data/README.md](data/README.md)を参照してください。Hugging Face / ModelScopeのハブ上のデータセットを使用するか、ローカルディスクのデータセットをロードしてください。

> [!NOTE]
> カスタムデータセットを使用するように `data/dataset_info.json` を更新してください。

### クイックスタート

Llama3-8B-InstructモデルのLoRAの **ファインチューニング**、**推論**、**マージ** を実行するには、それぞれ以下の3つのコマンドを使用します。

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

高度な使い方（分散トレーニングを含む）については[examples/README.md](examples/README.md)を参照してください。

> [!TIP]
> ヘルプ情報を表示するには `llamafactory-cli help` を使用します。

### LLaMAボードGUIによるファインチューニング（Powered by [Gradio](https://github.com/gradio-app/gradio)

```bash
llamafactory-cli webui
```

### Dockerのビルド

CUDAユーザー向け:

```bash
cd docker/docker-cuda/
docker-compose up -d
docker-compose exec llamafactory bash
```

アセンドNPUユーザー向け:

```bash
cd docker/docker-npu/
docker-compose up -d
docker-compose exec llamafactory bash
```

<details><summary>Docker Composeを使わないビルド</summary>

CUDAユーザー向け:

```bash
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg INSTALL_BNB=false \
    --build-arg INSTALL_VLLM=false \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg INSTALL_FLASHATTN=false \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llamafactory:latest .

docker run -dit --gpus=all \
    -v ./hf_cache:/root/.cache/huggingface \
    -v ./ms_cache:/root/.cache/modelscope \
    -v ./data:/app/data \
    -v ./output:/app/output \
    -p 7860:7860 \
    -p 8000:8000 \
    --shm-size 16G \
    --name llamafactory \
    llamafactory:latest

docker exec -it llamafactory bash
```

アセンドNPUユーザー向け:

```bash
# あなたの環境に合わせてdockerイメージを選択
docker build -f ./docker/docker-npu/Dockerfile \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    -t llamafactory:latest .

# リソースに応じて`device`を変更
docker run -dit \
    -v ./hf_cache:/root/.cache/huggingface \
    -v ./ms_cache:/root/.cache/modelscope \
    -v ./data:/app/data \
    -v ./output:/app/output \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -p 7860:7860 \
    -p 8000:8000 \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    --shm-size 16G \
    --name llamafactory \
    llamafactory:latest

docker exec -it llamafactory bash
```

</details>

<details><summary>ボリュームの詳細</summary>

- hf_cache: ホストマシンのHugging Faceキャッシュを利用する。別のディレクトリに既にキャッシュが存在する場合は、再割り当て可能。
- data: LLaMA Board GUIで選択できるように、データセットをホストマシンのこのディレクトリに置く。
- output: エクスポート先をこの場所に設定することで、マージされた結果にホストマシンから直接アクセスできるようになる。

</details>

### OpenAIスタイルのAPIとvLLMを使ったデプロイ

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml
```

> [!TIP]
> APIドキュメントは https://platform.openai.com/docs/api-reference/chat/create。

### ModelScope Hubからダウンロード

Hugging Faceからのモデルやデータセットのダウンロードに問題がある場合は、ModelScopeをご利用ください。

```bash
export USE_MODELSCOPE_HUB=1 # Windowsの場合、`set USE_MODELSCOPE_HUB=1`
```

ModelScope HubのモデルIDを `model_name_or_path` に指定してモデルをトレーニングする。モデルIDの完全なリストは[ModelScope Hub](https://modelscope.cn/models)にあります。例えば、`LLM-Research/Meta-Llama-3-8B-Instruct`です。

### W&Bロガーの使用

実験結果のロギングに[Weights & Biases](https://wandb.ai)を使用するには、yamlファイルに以下の引数を追加する必要があります。

```yaml
report_to: wandb
run_name: test_run # オプション
```

W&Bアカウントでログインするためにトレーニングタスクを起動する際に、`WANDB_API_KEY`を[あなたのキー](https://wandb.ai/authorize)に設定します。

## LLaMAファクトリーを使用したプロジェクト

取り込むべきプロジェクトがありましたら、メールでご連絡いただくか、プルリクエストを作成してください。

<details><summary>クリックで表示</summary>

1. Wang et al. ESRL: Efficient Sampling-based Reinforcement Learning for Sequence Generation. 2023. [[arxiv]](https://arxiv.org/abs/2308.02223)
1. Yu et al. Open, Closed, or Small Language Models for Text Classification? 2023. [[arxiv]](https://arxiv.org/abs/2308.10092)
1. Wang et al. UbiPhysio: Support Daily Functioning, Fitness, and Rehabilitation with Action Understanding and Feedback in Natural Language. 2023. [[arxiv]](https://arxiv.org/abs/2308.10526)
1. Luceri et al. Leveraging Large Language Models to Detect Influence Campaigns in Social Media. 2023. [[arxiv]](https://arxiv.org/abs/2311.07816)
1. Zhang et al. Alleviating Hallucinations of Large Language Models through Induced Hallucinations. 2023. [[arxiv]](https://arxiv.org/abs/2312.15710)
1. Wang et al. Know Your Needs Better: Towards Structured Understanding of Marketer Demands with Analogical Reasoning Augmented LLMs. KDD 2024. [[arxiv]](https://arxiv.org/abs/2401.04319)
1. Wang et al. CANDLE: Iterative Conceptualization and Instantiation Distillation from Large Language Models for Commonsense Reasoning. ACL 2024. [[arxiv]](https://arxiv.org/abs/2401.07286)
1. Choi et al. FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs. 2024. [[arxiv]](https://arxiv.org/abs/2402.05904)
1. Zhang et al. AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts. 2024. [[arxiv]](https://arxiv.org/abs/2402.07625)
1. Lyu et al. KnowTuning: Knowledge-aware Fine-tuning for Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.11176)
1. Yang et al. LaCo: Large Language Model Pruning via Layer Collaps. 2024. [[arxiv]](https://arxiv.org/abs/2402.11187)
1. Bhardwaj et al. Language Models are Homer Simpson! Safety Re-Alignment of Fine-tuned Language Models through Task Arithmetic. 2024. [[arxiv]](https://arxiv.org/abs/2402.11746)
1. Yang et al. Enhancing Empathetic Response Generation by Augmenting LLMs with Small-scale Empathetic Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.11801)
1. Yi et al. Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding. ACL 2024 Findings. [[arxiv]](https://arxiv.org/abs/2402.11809)
1. Cao et al. Head-wise Shareable Attention for Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.11819)
1. Zhang et al. Enhancing Multilingual Capabilities of Large Language Models through Self-Distillation from Resource-Rich Languages. 2024. [[arxiv]](https://arxiv.org/abs/2402.12204)
1. Kim et al. Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2402.14714)
1. Yu et al. KIEval: A Knowledge-grounded Interactive Evaluation Framework for Large Language Models. ACL 2024. [[arxiv]](https://arxiv.org/abs/2402.15043)
1. Huang et al. Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning. 2024. [[arxiv]](https://arxiv.org/abs/2403.02333)
1. Duan et al. Negating Negatives: Alignment without Human Positive Samples via Distributional Dispreference Optimization. 2024. [[arxiv]](https://arxiv.org/abs/2403.03419)
1. Xie and Schwertfeger. Empowering Robotics with Large Language Models: osmAG Map Comprehension with LLMs. 2024. [[arxiv]](https://arxiv.org/abs/2403.08228)
1. Wu et al. Large Language Models are Parallel Multilingual Learners. 2024. [[arxiv]](https://arxiv.org/abs/2403.09073)
1. Zhang et al. EDT: Improving Large Language Models' Generation by Entropy-based Dynamic Temperature Sampling. 2024. [[arxiv]](https://arxiv.org/abs/2403.14541)
1. Weller et al. FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions. 2024. [[arxiv]](https://arxiv.org/abs/2403.15246)
1. Hongbin Na. CBT-LLM: A Chinese Large Language Model for Cognitive Behavioral Therapy-based Mental Health Question Answering. COLING 2024. [[arxiv]](https://arxiv.org/abs/2403.16008)
1. Zan et al. CodeS: Natural Language to Code Repository via Multi-Layer Sketch. 2024. [[arxiv]](https://arxiv.org/abs/2403.16443)
1. Liu et al. Extensive Self-Contrast Enables Feedback-Free Language Model Alignment. 2024. [[arxiv]](https://arxiv.org/abs/2404.00604)
1. Luo et al. BAdam: A Memory Efficient Full Parameter Training Method for Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2404.02827)
1. Du et al. Chinese Tiny LLM: Pretraining a Chinese-Centric Large Language Model. 2024. [[arxiv]](https://arxiv.org/abs/2404.04167)
1. Ma et al. Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens Rotation. ICML 2024. [[arxiv]](https://arxiv.org/abs/2404.04316)
1. Liu et al. Dynamic Generation of Personalities with Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2404.07084)
1. Shang et al. How Far Have We Gone in Stripped Binary Code Understanding Using Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2404.09836)
1. Huang et al. LLMTune: Accelerate Database Knob Tuning with Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2404.11581)
1. Deng et al. Text-Tuple-Table: Towards Information Integration in Text-to-Table Generation via Global Tuple Extraction. 2024. [[arxiv]](https://arxiv.org/abs/2404.14215)
1. Acikgoz et al. Hippocrates: An Open-Source Framework for Advancing Large Language Models in Healthcare. 2024. [[arxiv]](https://arxiv.org/abs/2404.16621)
1. Zhang et al. Small Language Models Need Strong Verifiers to Self-Correct Reasoning. ACL 2024 Findings. [[arxiv]](https://arxiv.org/abs/2404.17140)
1. Zhou et al. FREB-TQA: A Fine-Grained Robustness Evaluation Benchmark for Table Question Answering. NAACL 2024. [[arxiv]](https://arxiv.org/abs/2404.18585)
1. Xu et al. Large Language Models for Cyber Security: A Systematic Literature Review. 2024. [[arxiv]](https://arxiv.org/abs/2405.04760)
1. Dammu et al. "They are uncultured": Unveiling Covert Harms and Social Threats in LLM Generated Conversations. 2024. [[arxiv]](https://arxiv.org/abs/2405.05378)
1. Yi et al. A safety realignment framework via subspace-oriented model fusion for large language models. 2024. [[arxiv]](https://arxiv.org/abs/2405.09055)
1. Lou et al. SPO: Multi-Dimensional Preference Sequential Alignment With Implicit Reward Modeling. 2024. [[arxiv]](https://arxiv.org/abs/2405.12739)
1. Zhang et al. Getting More from Less: Large Language Models are Good Spontaneous Multilingual Learners. 2024. [[arxiv]](https://arxiv.org/abs/2405.13816)
1. Zhang et al. TS-Align: A Teacher-Student Collaborative Framework for Scalable Iterative Finetuning of Large Language Models. 2024. [[arxiv]](https://arxiv.org/abs/2405.20215)
1. Zihong Chen. Sentence Segmentation and Sentence Punctuation Based on XunziALLM. 2024. [[paper]](https://aclanthology.org/2024.lt4hala-1.30)
1. Gao et al. The Best of Both Worlds: Toward an Honest and Helpful Large Language Model. 2024. [[arxiv]](https://arxiv.org/abs/2406.00380)
1. Wang and Song. MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset. 2024. [[arxiv]](https://arxiv.org/abs/2406.02106)
1. Hu et al. Computational Limits of Low-Rank Adaptation (LoRA) for Transformer-Based Models. 2024. [[arxiv]](https://arxiv.org/abs/2406.03136)
1. Ge et al. Time Sensitive Knowledge Editing through Efficient Finetuning. ACL 2024. [[arxiv]](https://arxiv.org/abs/2406.04496)
1. Tan et al. Peer Review as A Multi-Turn and Long-Context Dialogue with Role-Based Interactions. 2024. [[arxiv]](https://arxiv.org/abs/2406.05688)
1. Song et al. Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters. 2024. [[arxiv]](https://arxiv.org/abs/2406.05955)
1. Gu et al. RWKV-CLIP: A Robust Vision-Language Representation Learner. 2024. [[arxiv]](https://arxiv.org/abs/2406.06973)
1. Chen et al. Advancing Tool-Augmented Large Language Models: Integrating Insights from Errors in Inference Trees. 2024. [[arxiv]](https://arxiv.org/abs/2406.07115)
1. Zhu et al. Are Large Language Models Good Statisticians?. 2024. [[arxiv]](https://arxiv.org/abs/2406.07815)
1. Li et al. Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning. 2024. [[arxiv]](https://arxiv.org/abs/2406.10099)
1. Ding et al. IntentionQA: A Benchmark for Evaluating Purchase Intention Comprehension Abilities of Language Models in E-commerce. 2024. [[arxiv]](https://arxiv.org/abs/2406.10173)
1. He et al. COMMUNITY-CROSS-INSTRUCT: Unsupervised Instruction Generation for Aligning Large Language Models to Online Communities. 2024. [[arxiv]](https://arxiv.org/abs/2406.12074)
1. Lin et al. FVEL: Interactive Formal Verification Environment with Large Language Models via Theorem Proving. 2024. [[arxiv]](https://arxiv.org/abs/2406.14408)
1. Treutlein et al. Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data. 2024. [[arxiv]](https://arxiv.org/abs/2406.14546)
1. Feng et al. SS-Bench: A Benchmark for Social Story Generation and Evaluation. 2024. [[arxiv]](https://arxiv.org/abs/2406.15695)
1. Feng et al. Self-Constructed Context Decompilation with Fined-grained Alignment Enhancement. 2024. [[arxiv]](https://arxiv.org/abs/2406.17233)
1. Liu et al. Large Language Models for Cuffless Blood Pressure Measurement From Wearable Biosignals. 2024. [[arxiv]](https://arxiv.org/abs/2406.18069)
1. Iyer et al. Exploring Very Low-Resource Translation with LLMs: The University of Edinburgh’s Submission to AmericasNLP 2024 Translation Task. AmericasNLP 2024. [[paper]](https://aclanthology.org/2024.americasnlp-1.25)
1. **[StarWhisper](https://github.com/Yu-Yang-Li/StarWhisper)**: ChatGLM2-6BとQwen-14Bをベースにした天文学用の大規模言語モデル。
1. **[DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)**: Baichuan-13Bをベースとした中国法領域に特化した大規模言語モデルは、法知識の検索と推論が可能である。
1. **[Sunsimiao](https://github.com/X-D-Lab/Sunsimiao)**: Baichuan-7BとChatGLM-6Bをベースにした、中国語医療分野に特化した大規模言語モデル。
1. **[CareGPT](https://github.com/WangRongsheng/CareGPT)**: LLaMA2-7BとBaichuan-13Bをベースとした中国語医療分野の大規模言語モデルシリーズ。
1. **[MachineMindset](https://github.com/PKU-YuanGroup/Machine-Mindset/)**: MBTIパーソナリティ大規模言語モデルのシリーズで、異なるデータセットとトレーニング方法に基づいて、16の異なるパーソナリティタイプをLLMに与えることができる。
1. **[Luminia-13B-v3](https://huggingface.co/Nekochu/Luminia-13B-v3)**: メタデータ生成に特化した大規模言語モデルで安定した普及を目指す。[[🤗デモ]](https://huggingface.co/spaces/Nekochu/Luminia-13B_SD_Prompt)
1. **[Chinese-LLaVA-Med](https://github.com/BUAADreamer/Chinese-LLaVA-Med)**: LLaVA-1.5-7Bをベースとした、中国語医療領域に特化したマルチモーダル大規模言語モデル。
1. **[AutoRE](https://github.com/THUDM/AutoRE)**: 大規模言語モデルに基づく文書レベル関係抽出システム。
1. **[NVIDIA RTX AI Toolkit](https://github.com/NVIDIA/RTX-AI-Toolkit)**: NVIDIA RTX用のWindows PC上でLLMを微調整するためのSDK。
1. **[LazyLLM](https://github.com/LazyAGI/LazyLLM)**: マルチエージェントLLMアプリケーションを簡単かつ容易に構築でき、LLaMAファクトリーによるモデルのファインチューニングをサポートします。

</details>

## ライセンス

このリポジトリは[Apache-2.0 License](LICENSE)の下でライセンスされています。

対応するモデルのウェイトを使用するには、モデルライセンスに従ってください: [Baichuan 2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/Community%20License%20for%20Baichuan%202%20Model.pdf) / [BLOOM](https://huggingface.co/spaces/bigscience/license) / [ChatGLM3](https://github.com/THUDM/ChatGLM3/blob/main/MODEL_LICENSE) / [Command R](https://cohere.com/c4ai-cc-by-nc-license) / [DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/LICENSE-MODEL) / [Falcon](https://huggingface.co/tiiuae/falcon-180B/blob/main/LICENSE.txt) / [Gemma](https://ai.google.dev/gemma/terms) / [GLM-4](https://huggingface.co/THUDM/glm-4-9b/blob/main/LICENSE) / [InternLM2](https://github.com/InternLM/InternLM#license) / [Llama](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) / [Llama 2 (LLaVA-1.5)](https://ai.meta.com/llama/license/) / [Llama 3](https://llama.meta.com/llama3/license/) / [Mistral](LICENSE) / [OLMo](LICENSE) / [Phi-1.5/Phi-2](https://huggingface.co/microsoft/phi-1_5/resolve/main/Research%20License.docx) / [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE) / [Qwen](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) / [StarCoder 2](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) / [XVERSE](https://github.com/xverse-ai/XVERSE-13B/blob/main/MODEL_LICENSE.pdf) / [Yi](https://huggingface.co/01-ai/Yi-6B/blob/main/LICENSE) / [Yi-1.5](LICENSE) / [Yuan 2](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/LICENSE-Yuan)

## 引用

もしこの研究がお役に立つようでしたら、以下のように引用してください:

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```

## 謝辞

このリポジトリは、[PEFT](https://github.com/huggingface/peft)、[TRL](https://github.com/huggingface/trl)、[QLoRA](https://github.com/artidoro/qlora)、[FastChat](https://github.com/lm-sys/FastChat)の恩恵を受けています。彼らの素晴らしい仕事に感謝します。

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/LLaMA-Factory&type=Date)
