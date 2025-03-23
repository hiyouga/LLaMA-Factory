import json
import logging
import sys
import time

import fire


try:
    import jieba
    from datasets import load_dataset
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_chinese import Rouge
except Exception as err:
    print(
        f"Failed to start, please install these requirements:\n{err}\n\t pip install jieba jsonlines nltk rouge_chinese datasets"
    )
    sys.exit(1)

try:
    # 关掉输出
    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()
except Exception as err:
    print(f"failed to initialize jieba, exiting,\n{err}")


def compute_metrics(sample):
    hypothesis = list(jieba.cut(sample["predict"]))
    reference = list(jieba.cut(sample["label"]))

    bleu_score = sentence_bleu(
        [list(sample["label"])],
        list(sample["predict"]),
        smoothing_function=SmoothingFunction().method3,
    )

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    metric_result = {}
    for k, v in result.items():
        metric_result[k] = round(v["f"] * 100, 4)
    metric_result["bleu-4"] = round(bleu_score * 100, 4)

    return metric_result


def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")

    # 多进程
    dataset = dataset.map(compute_metrics, num_proc=6, remove_columns=dataset.column_names)
    score_dict = dataset.to_dict()

    average_score = {}
    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        print(f"{task}: {sum(scores) / len(scores):.4f}")
        average_score[task] = sum(scores) / len(scores)

    json.dump(
        average_score,
        open("predictions_score.json", "w+", encoding="utf-8"),
        indent=4,
    )
    print(f"\nDone in {time.time() - start_time: .3f}s \nscore file saved to predictions_score.json")
    # return average_score


if __name__ == "__main__":
    fire.Fire(main)
