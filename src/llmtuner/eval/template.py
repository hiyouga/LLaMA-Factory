from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from ..data import Role
from ..extras.constants import CHOICES


if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class EvalTemplate:
    """
    Represents an evaluation template for generating prompts and responses.

    Attributes
    ----------
    system : str
        The system information placeholder.

    choice : str
        The choice placeholder.

    answer : str
        The answer placeholder.

    prefix : str
        The prefix placeholder.

    Methods
    -------
    parse_example(example: Dict[str, str]) -> Tuple[str, str]:
        Parses an example dictionary into a prompt-response pair.

    format_example(target_data: Dict[str, str], support_set: "Dataset", subject_name: str) -> List[Dict[str, str]]:
        Formats an example with a support set for evaluation.
    """
    system: str
    choice: str
    answer: str
    prefix: str

    def parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        """
        Parse an example dictionary into a prompt-response pair.

        Parameters
        ----------
        example : Dict[str, str]
            The example dictionary.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the parsed prompt and response.
        """
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self, target_data: Dict[str, str], support_set: "Dataset", subject_name: str
    ) -> List[Dict[str, str]]:
        """
        Format an example with a support set for evaluation.

        Parameters
        ----------
        target_data : Dict[str, str]
            The target data dictionary.

        support_set : "Dataset"
            The support dataset.

        subject_name : str
            The name of the subject.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries containing formatted messages.
        """
        messages = []
        for k in range(len(support_set)):
            prompt, response = self.parse_example(support_set[k])
            messages.append({"role": Role.USER, "content": prompt})
            messages.append({"role": Role.ASSISTANT, "content": response})

        prompt, response = self.parse_example(target_data)
        messages.append({"role": Role.USER, "content": prompt})
        messages.append({"role": Role.ASSISTANT, "content": response})
        messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages


eval_templates: Dict[str, "EvalTemplate"] = {}


def register_eval_template(name: str, system: str, choice: str, answer: str, prefix: str) -> None:
    """
    Register an evaluation template.

    Parameters
    ----------
    name : str
        The name of the evaluation template.

    system : str
        The system information placeholder.

    choice : str
        The choice placeholder.

    answer : str
        The answer placeholder.

    prefix : str
        The prefix placeholder.

    Returns
    -------
    None
    """
    eval_templates[name] = EvalTemplate(system=system, choice=choice, answer=answer, prefix=prefix)


def get_eval_template(name: str) -> "EvalTemplate":
    """
    Get an evaluation template by name.

    Parameters
    ----------
    name : str
        The name of the evaluation template.

    Returns
    -------
    "EvalTemplate"
        The evaluation template.
    """
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template


register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)


register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix="\n",
)
