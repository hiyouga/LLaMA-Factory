#!/usr/bin/env python3
"""
Test script to debug data preprocessing and padding behavior.
This will help us understand why max_length padding isn't working correctly.
"""

import os
import sys
sys.path.insert(0, 'src')

from llamafactory.hparams import get_train_args
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer

def test_data_preprocessing():
    """Test data preprocessing with different padding configurations."""
    
    # Test configurations
    configs = [
        {
            "name": "max_length_padding", 
            "pad_to_multiple_of": "max_length",
            "cutoff_len": 4096
        }
    ]
    
    base_args = [
        "--model_name_or_path", "Qwen/Qwen2.5-7B-Instruct",
        "--stage", "sft",
        "--do_train", "true", 
        "--dataset", "tbench_traces_sharegptv1",
        "--template", "qwen",
        "--finetuning_type", "full",
        "--output_dir", "test_output",
        "--overwrite_output_dir", "true",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "1e-5",
        "--num_train_epochs", "1",
        "--fp16", "false",
        "--max_samples", "10",  # Small sample for testing
        "--logging_steps", "1"
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing {config['name']}")
        print(f"{'='*60}")
        
        # Create args for this configuration
        test_args = base_args + [
            "--cutoff_len", str(config["cutoff_len"]),
            "--pad_to_multiple_of", str(config["pad_to_multiple_of"])
        ]
        
        try:
            # Get training arguments
            model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(test_args)
            
            print(f"Configuration:")
            print(f"  - cutoff_len: {data_args.cutoff_len}")
            print(f"  - pad_to_multiple_of: {data_args.pad_to_multiple_of}")
            
            # Load tokenizer
            print(f"Loading tokenizer...")
            tokenizer_module = load_tokenizer(model_args)
            tokenizer = tokenizer_module["tokenizer"]
            
            # Get template
            template = get_template_and_fix_tokenizer(tokenizer, data_args)
            
            # Load dataset
            print(f"Loading dataset...")
            dataset_module = get_dataset(template, model_args, data_args, training_args, "sft", tokenizer)
            train_dataset = dataset_module["train_dataset"]
            
            print(f"Dataset loaded: {len(train_dataset)} samples")
            
            # Test first few samples
            print(f"\nSample analysis:")
            for i in range(min(5, len(train_dataset))):
                sample = train_dataset[i]
                input_ids = sample["input_ids"]
                labels = sample.get("labels", sample.get("input_ids"))
                
                print(f"  Sample {i}:")
                print(f"    input_ids length: {len(input_ids)}")
                print(f"    labels length: {len(labels) if labels is not None else 'None'}")
                print(f"    input_ids type: {type(input_ids)}")
                
                # Check if all samples have same length (for max_length padding)
                if config["pad_to_multiple_of"] == "max_length":
                    if len(input_ids) != config["cutoff_len"]:
                        print(f"    ❌ Length mismatch! Expected {config['cutoff_len']}, got {len(input_ids)}")
                    else:
                        print(f"    ✅ Length consistent with cutoff_len: {len(input_ids)}")
            
            # Test the data collator behavior
            print(f"\nTesting data collator behavior:")
            from llamafactory.data import SFTDataCollatorWith4DAttentionMask
            from llamafactory.train.trainer_utils import get_optimal_pad_multiple
            
            # Get the optimal padding (simulating what happens in workflow)
            optimal_padding = get_optimal_pad_multiple(None, model_args, data_args, training_args)  
            print(f"  optimal_padding detected: {optimal_padding}")
            
            # Create collator  
            data_collator = SFTDataCollatorWith4DAttentionMask(
                template=template,
                model=None,  # We don't need the model for this test
                pad_to_multiple_of=optimal_padding,
                **tokenizer_module,
            )
            
            # Test batch with 3 samples
            test_batch = [train_dataset[i] for i in range(min(3, len(train_dataset)))]
            print(f"  Input samples lengths: {[len(s['input_ids']) for s in test_batch]}")
            
            try:
                collated_batch = data_collator(test_batch)
                print(f"  Collated batch shape: {collated_batch['input_ids'].shape}")
                print(f"  All sequences same length: {len(set([len(row) for row in collated_batch['input_ids']]))==1}")
            except Exception as e:
                print(f"  ❌ Collator error: {e}")
                        
        except Exception as e:
            print(f"❌ Error testing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing data preprocessing and padding behavior...")
    test_data_preprocessing()
    print("\nDone!")