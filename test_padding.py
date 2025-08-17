#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from llamafactory.hparams import DataArguments
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from transformers import AutoTokenizer
import torch

def test_force_padding():
    print("Testing force_sequence_length_padding feature...")
    
    # Setup
    cutoff_len = 128
    data_args = DataArguments(
        force_sequence_length_padding=True,
        cutoff_len=cutoff_len,
        train_on_prompt=False
    )
    
    # Use a simple tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token
    
    template = get_template_and_fix_tokenizer("default", tokenizer)
    
    processor = SupervisedDatasetProcessor(
        tokenizer=tokenizer,
        data_args=data_args,
        template=template,
        processor=None
    )
    
    # Create test data - simple conversation
    examples = {
        "_prompt": [
            [{"role": "user", "content": "Hello"}]
        ],
        "_response": [
            [{"role": "assistant", "content": "Hi there!"}]
        ],
        "_system": [None],
        "_tools": [None],
        "_images": [None],
        "_videos": [None],
        "_audios": [None],
    }
    
    # Process the dataset
    result = processor.preprocess_dataset(examples)
    
    # Check results
    print(f"Number of processed examples: {len(result['input_ids'])}")
    
    if len(result['input_ids']) > 0:
        input_ids = result['input_ids'][0]
        labels = result['labels'][0]
        attention_mask = result['attention_mask'][0]
        
        print(f"Input length: {len(input_ids)}")
        print(f"Expected length: {cutoff_len}")
        print(f"Labels length: {len(labels)}")
        print(f"Attention mask length: {len(attention_mask)}")
        
        # Verify padding
        if len(input_ids) == cutoff_len:
            print("✅ Padding successful - sequence is exactly cutoff_len")
        else:
            print(f"❌ Padding failed - expected {cutoff_len}, got {len(input_ids)}")
            
        # Check if padding tokens are present
        pad_count = sum(1 for token_id in input_ids if token_id == tokenizer.pad_token_id)
        print(f"Number of padding tokens: {pad_count}")
        
        # Check that labels are properly padded with IGNORE_INDEX
        ignore_count = sum(1 for label in labels if label == -100)
        print(f"Number of IGNORE_INDEX labels: {ignore_count}")
        
        return len(input_ids) == cutoff_len
    else:
        print("❌ No examples processed")
        return False

if __name__ == "__main__":
    success = test_force_padding()
    sys.exit(0 if success else 1)