import json
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset as Dataset_torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor


class ImageCaptioningDataset(Dataset_torch):
    def __init__(self, dataset: Dataset, image_path: str, processor: AutoProcessor):
        self.processor = processor
        self.dataset = dataset
        self.image_path = image_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source = self.dataset[idx]
        image_id = source['image']
        image = Image.open(os.path.join(self.image_path, image_id))
        convs = source['conversations']
        prompt = convs[0]['value']
        label = convs[1]['value']
        image_inputs = self.processor(image, return_tensors="pt")
        image_inputs = {k: v.squeeze() for k, v in image_inputs.items()}
        inputs = {
            "input_ids": prompt,
            "labels": label,
        }
        for key in image_inputs:
            inputs[key] = image_inputs[key]
        return inputs


@dataclass
class DataCollatorForVis2Seq:
    processor: AutoProcessor
    use_qformer: bool = False

    def __call__(self, features, return_tensors=None):
        processed_batch = {}
        for key in features[0].keys():
            if key == 'pixel_values':
                processed_batch[key] = torch.stack([example[key] for example in features])
            elif key == 'input_ids':
                text_inputs = self.processor.tokenizer(
                    [example[key] for example in features], padding="max_length", return_tensors="pt",
                    max_length=512,
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
                if self.use_qformer:
                    qformer_text_inputs = self.processor.qformer_tokenizer(
                        [example[key] for example in features], padding="max_length", return_tensors="pt",
                        max_length=512,
                    )
                    processed_batch["qformer_input_ids"] = qformer_text_inputs["input_ids"]
                    processed_batch["qformer_attention_mask"] = qformer_text_inputs["attention_mask"]
            elif key == 'labels':
                text_inputs = self.processor.tokenizer(
                    [example[key] for example in features], padding="max_length", return_tensors="pt",
                    max_length=512,
                )
                processed_batch["labels"] = text_inputs["input_ids"]
        return processed_batch
