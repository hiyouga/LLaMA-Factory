import os
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from transformers.trainer import PredictionOutput
from transformers.tokenization_utils import PreTrainedTokenizer

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from .peft_trainer import PeftTrainer

from .other import get_logger, IGNORE_INDEX


logger = get_logger(__name__)


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        for pred, label in zip(preds, labels):
            pred = pred[len(label) - np.sum(label == IGNORE_INDEX) : len(pred) - np.sum(pred == IGNORE_INDEX)] # remove prompts
            label = label[:len(label) - np.sum(label == IGNORE_INDEX)]

            hypothesis = list(jieba.cut(self.tokenizer.decode(pred, skip_special_tokens=True)))
            reference = list(jieba.cut(self.tokenizer.decode(label, skip_special_tokens=True)))

            if len(" ".join(hypothesis).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}


class Seq2SeqPeftTrainer(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def save_predictions(
            self,
            predict_results: PredictionOutput
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(predict_results.predictions, predict_results.label_ids):
                pred = pred[len(label) - np.sum(label == IGNORE_INDEX) : len(pred) - np.sum(pred == IGNORE_INDEX)] # remove prompts
                label = label[:len(label) - np.sum(label == IGNORE_INDEX)]

                pred = self.tokenizer.decode(pred, skip_special_tokens=True)
                label = self.tokenizer.decode(label, skip_special_tokens=True)

                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
