from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from llmtuner.extras.constants import CHOICES

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class EvalTemplate:

    system: str
    choice: str
    answer: str
    prefix: str

    def parse_example(
        self,
        example: Dict[str, str]
    ) -> Tuple[str, str]:
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self,
        target_data: Dict[str, str],
        support_set: "Dataset",
        subject_name: str,
        use_history: bool
    ) -> Tuple[str, str, List[Tuple[str, str]]]:
        query, resp = self.parse_example(target_data)
        history = [self.parse_example(support_set[k]) for k in range(len(support_set))]

        if len(history):
            temp = history.pop(0)
            history.insert(0, (self.system.format(subject=subject_name) + temp[0], temp[1]))
        else:
            query = self.system.format(subject=subject_name) + query

        if not use_history:
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        return query.strip(), resp, history


eval_templates: Dict[str, EvalTemplate] = {}


def register_eval_template(
    name: str,
    system: str,
    choice: str,
    answer: str,
    prefix: str
) -> None:
    eval_templates[name] = EvalTemplate(
        system=system,
        choice=choice,
        answer=answer,
        prefix=prefix
    )


def get_eval_template(name: str) -> EvalTemplate:
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" "
)


register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix="\n"
)
