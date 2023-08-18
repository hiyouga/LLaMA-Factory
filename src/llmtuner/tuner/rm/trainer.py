import os
import json
import torch
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core.trainer import PeftTrainer

if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput
    from transformers.modeling_utils import PreTrainedModel


logger = get_logger(__name__)


class PairwisePeftTrainer(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute pairwise loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True # override property to return eval_loss

    def compute_loss(
        self,
        model: "PreTrainedModel",
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes pairwise loss. The first n examples are chosen and the last n examples are rejected.

        We use score on the EOS token to represent reward of the whole sentence.

        Subclass and override to inject custom behavior. It should not be directly used by external scripts.

        Note that the first element will be removed from the output tuple.

        See: https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/trainer.py#L3509
        """
        batch_size = inputs["input_ids"].size(0) // 2
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)
        if values.size(0) != inputs["input_ids"].size(0): # adapt to chatglm2
            values = torch.transpose(values, 0, 1)
        r_accept, r_reject = values[:, -1].split(batch_size, dim=0)
        loss = -torch.log(torch.sigmoid(r_accept - r_reject)).mean()
        return (loss, [loss, r_accept, r_reject]) if return_outputs else loss

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        acc_scores, rej_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for acc_score, rej_score in zip(acc_scores, rej_scores):
                res.append(json.dumps({"accept": round(float(acc_score), 2), "reject": round(float(rej_score), 2)}))
            writer.write("\n".join(res))
