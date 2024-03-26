from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..data import Role
from ..extras.constants import CHOICES


@dataclass
class EvalTemplate:
    system: str
    choice: str
    answer: str
    prefix: str

    def _parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response)
        """
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], subject_name: str
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        for k in range(len(support_set)):
            prompt, response = self._parse_example(support_set[k])
            messages.append({"role": Role.USER.value, "content": prompt})
            messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data)
        messages.append({"role": Role.USER.value, "content": prompt})
        messages.append({"role": Role.ASSISTANT.value, "content": response})
        messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages


eval_templates: Dict[str, "EvalTemplate"] = {}


def _register_eval_template(name: str, system: str, choice: str, answer: str, prefix: str) -> None:
    eval_templates[name] = EvalTemplate(system=system, choice=choice, answer=answer, prefix=prefix)


def get_eval_template(name: str) -> "EvalTemplate":
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)


_register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix=" ",
)
