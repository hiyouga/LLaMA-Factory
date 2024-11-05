# bookIPS-Solvook-LLM (Edu-Llama)

> [TIPS] 교육콘텐츠IP 라이선싱 플랫폼 고도화를 위한 컴포넌트 단위 콘텐츠 관계성 분석 AI기술개발
> [TIPS] AI Technology Development for Component-Level Content Relationship Analysis to Enhance Educational Content IP Licensing Platform

Developing a LLM based on LLaMA 3.1 for tasks related to korean - english educational contents.

<br>

----
# 1. Setup and Preparation
## get Liscense
Our model is based on LLaMA 3.1 8B Instruct, for using llama 3.1 edu model you have to get liscense and access token
(`huggingface-cli login` needs access token)

https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Setup
```bash
git clone --recurse-submodules https://github.com/MLAI-Yonsei/bookIPs-Solvook-LLM.git
cd bookIPs-Solvook-LLM
conda create -n <name> python==3.11
conda activate <name>
pip install -e ".[torch,metrics]"
pip install --upgrade huggingface_hub
huggingface-cli login
pip install -r requirements_rag.txt
```
> [!TIP]
Use `pip install --no-deps -e .` to resolve package conflicts.

----



<br>


# 2. Fine-tuning Edu-Llama

## Data preparation
* Download pre-procssed dataset in [link](https://drive.google.com/drive/folders/1Y74O82TE-1f_6bq8w2r7JWjxdj7i42Rw?usp=drive_link) and place the provided data in the `/data` folder.
    * If you want to pre-process dataset by yourself, run `generate_instruction_set.py`
    ```
    generate_instruction_set.py --working_dir=<DATA_PATH>
    ```
    * ```DATA_PATH``` &mdash; The path where 'Solvook_handout_DB_english.xlsx' exists.

*  The data information has been entered in [data/dataset_info.json](data/dataset_info.json), so please place the data files as they are without modifying them.


## Model preparation
* You can simply download fine-tuned model in [link](https://drive.google.com/drive/folders/1RgNjZTlmye6kt4I0o6T-z9KIYCUzq3vo?usp=drive_link) and place the provided adapter model files in a specific folder. The location of the adapter model files doesn’t matter, but when performing inference, apply the adapter model file path to adapter_name_or_path. By default, it is set to **saves/llama3.1_edu**.
* If you want to fine-tune model by yourself, please refer to [Train](##train).


> [!Note]
> **Check whether your dataset and model path are on right place.**


## Train
* If you want to edit argument for training, edit file **examples/train_lora/llama3_lora_sft.yaml**

```bash
bash finetune.sh
```


> [!TIP]
> Attach `CUDA_VISIBLE_DEVICES=1,2,3` in front of code above to utilize multi GPU

> [!TIP]
> for more information to extend usage of LLaMA Factory package, visit https://github.com/hiyouga/LLaMA-Factory

----

<br>

# 3. Run RAG

## Pre-process dataset for RAG
* Download pre-processed dataset for RAG in [link](https://drive.google.com/drive/folders/1Y74O82TE-1f_6bq8w2r7JWjxdj7i42Rw?usp=drive_link).
    * Three files: `solvook_handout_tr.csv`, `solvook_handout_val.csv`, and `solvook_handout_te.csv`
    * If you want to pre-process dataset by yourself, run `preprocess_rag.py`
    ```
    preprocess_rag.py --data_path=<DATA_PATH>
    ```
    * ```DATA_PATH``` &mdash; The path where 'Solvook_handout_DB_english.xlsx' exists.


> [!Important]
> **We align the data split with instruction set and RAG set.**


## Set Vector DB
* To make and save vector DB, run `vector_db.py`:
    ```
    vector_db.py --query_path=<QUERY_PATH> --db_path=<DB_PATH> --openai_api_key=<API_KEY> \
                 --task=<TASK> --top_k=<TOP_K> --search_type=<SEARCH_TYPE>
    ```
    * ```QUERY_PATH``` &mdash; The path of '`solvook_handout_te.csv`' exists.
    * ```DB_PATH``` &mdash; The path of '`solvook_handout_tr.csv`' exists.
    * ```API_KEY``` &mdash; API key for openai
    * ```TASK``` &mdash; Designate task (Options: [**1**: Paragraph matching, **2**: Relation matching, **3**: Skill matching, **4**: Method matching])
    * ```TOP_K``` &mdash; The number of retrieved contents (Default: 6)
    * ```SEARCH_TYPE``` &mdash; The methods for calculation of similarity. (Options : [**sim**: Cosine similarity, **mmr**: Maximum marginal relevance search, **bm25**: BM25, **td_idf**: TF-IDF, **sim_bm25**: Ensemble of Cosine similarity and BM25]) (Default: mmr)



## Inference
### Inference with GPT-4o
* Assure to make vector DB, first. 
* RAG-based inference with GPT-4o can simply be done with `rag_gpt4o.py`:
    ```
    rag_gpt4o.py --query_path=<QUERY_PATH> --vector_db_path=<VECTOR_DB_PATH> \
                 --openai_api_key=<API_KEY> \
                 --temperature=<TEMPERATURE> \
                 --task=<TASK> [--in_context_sample] \
                 --result_path=<RESULT_PATH> \
                 [--ignore_wandb] \
                 --wandb_project=<WANDB_PROJECT> \
                 --wandb_entity=<WANDB_ENTITY>
    ```
    * ```QUERY_PATH``` &mdash; The path of '`solvook_handout_te.csv`' exists.
    * ```VECTOR_DB_PATH``` &mdash; The path of '`vector_db.json`' exists.
    * ```API_KEY``` &mdash; API key for openai
    * ```TEMPERATURE``` &mdash; Modulate the diversity of LLM output. Higher value allows more diverse output.
    * ```TASK``` &mdash; Designate task (Options: [**1**: Paragraph matching, **2**: Relation matching, **3**: Skill matching, **4**: Method matching])
    * ```--in_context_sample``` &mdash; Flag to put in-context sample for relation matching task.
    * ```RESULT_PATH``` &mdash; The path to save result.
    * ```--ignore_wandb``` &mdash; Flag to deactivate wandb
    * ```WANDB_PROJECT``` &mdash; Project name for wandb
    * ```WANDB_ENTITY``` &mdash; Entity name for wandb


### Inference with Edu-Llama
* For Edu-Llama, promptize is needed first with `edu_llama_promptize.py`:
    ```
    edu_llama_promptize.py --query_path=<QUERY_PATH> \
                 --vector_db_path=<VECTOR_DB_PATH> \
                 --split=<SPLIT> \
                 --task=<TASK>
    ```
    * ```QUERY_PATH``` &mdash; The path of '`solvook_handout_te.csv`' exists.
    * ```VECTOR_DB_PATH``` &mdash; The path of '`vector_db.json`' exists.
    * ```SPLIT``` &mdash; Choose which splits to load (Options: [**tr**: Training set, **val**: Validation set, **te**: Test set])
    * ```TASK``` &mdash; Designate task (Options: [**1**: Paragraph matching, **2**: Relation matching, **3**: Skill matching, **4**: Method matching])


* If you want to edit argument for training, edit file **examples/train_lora/llama3_lora_predict.yaml**

    ```bash
    bash inference.sh
    ```

---
<br>




# 4. For LLM evaluation in benchmark set
> [!Caution]
> Because of using different package, you have to seperate environment for fine tuning (inference) and benchmark dataset evaluation
```bash
conda deactivate
conda create -n <name> python==3.10
conda activate <name>
cd lm-evaluation-harness
pip install -e .
```

* edit `peft_path` from **llama3.1_edu_evaluation.sh** for your adpater path
* If you want to edit argument for evaluation, edit file **llama3.1_edu_evaluation.sh**

```bash
bash llama3.1_edu_evaluation.sh
```


# Acknowledgements
* We heavilty refered the code from [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory) and [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We appreciate the authors for sharing their code.