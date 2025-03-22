import json
import sys
import time


try:
    import jieba
    import jsonlines
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_chinese import Rouge
except Exception:
    print("Failed to start, please install these requirements:\n\t pip install jieba jsonlines nltk rouge_chinese")
    sys.exit(1)


def main(filename: str):
    start_time = time.time()
    predictions = []
    references = []

    # 读 generated_predictions.jsonl
    with jsonlines.open(filename) as reader:
        for obj in reader:
            predictions.append(obj["predict"])
            references.append(obj["label"])

    # 与 src/llamafactory/train/sft/metric.py 相同逻辑的评分
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for pred, label in zip(predictions, references):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
            result = {
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0},
            }
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    # 输出
    average_score = {}
    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        print(f"{task}: {sum(scores) / len(scores):.4f}")
        average_score[task] = sum(scores) / len(scores)

    json.dump(average_score, open("predictions_score.json", "w+", encoding="utf-8"), indent=4)
    print(f"\nDone in {time.time() - start_time: .3f}s \nscore file saved to predictions_score.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_bleu.py <input.jsonl>")
        sys.exit(1)

    main(sys.argv[1])
