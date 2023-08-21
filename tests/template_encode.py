# Test Template Encode
# Usage: python .\tests\template_encode.py --model_name_and_path D:\llm\chinese-alpaca-2-7b
#                                          --template llama2_zh --query 'how are you?'
#                                          --history '[[\"Hello!\",\"Hi，I am llama2.\"]]'

import sys
import fire
from typing import List, Optional, Tuple
from transformers import AutoTokenizer
sys.path.append("./src")
from llmtuner.extras.template import get_template_and_fix_tokenizer


def encode(
        model_name_and_path: str,
        template: str,
        query: str,
        resp: Optional[str] = "",
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_and_path,
        trust_remote_code=True
    )

    template = get_template_and_fix_tokenizer(template, tokenizer)

    encoded_pairs = template.encode_multiturn(tokenizer, query, resp, history, system)
    for prompt_ids, answer_ids in encoded_pairs:
        print("="*50)
        print("prompt_ids: {}, answer_ids: {}".format(prompt_ids, answer_ids))
        print("prompt decode: {}".format(tokenizer.decode(prompt_ids)))


if __name__ == '__main__':
    fire.Fire(encode)
