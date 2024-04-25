from dataclasses import dataclass
from transformers import AutoProcessor


@dataclass
class DataCollatorForVis2Seq:
    processor: AutoProcessor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            if len(example["images"]) > 1:
                raise ValueError("This collator only supports one image per example")
            messages = example["messages"]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(example["images"][0])

        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorForMLLM:
    processor: AutoProcessor

    def __call__(self, examples):
        print(examples[0].keys())
        print(examples[0]["input_ids"])
        batch = {}
        return batch
