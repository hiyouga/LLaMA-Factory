#!/usr/bin/env python3
"""
Script to download big-reasoning-traces dataset, extract first N records,
and upload to Hugging Face account.

Usage:
    python download_big_reasoning_traces.py --hf_username tech-tao --dataset_name big-reasoning-traces-100k --max_records 10000 --streaming
"""

import argparse
import logging
from typing import Optional
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_process_dataset(
    source_dataset: str = "allenai/big-reasoning-traces",
    subset: str = "DeepSeek",
    max_records: int = 100000,
    use_streaming: bool = False
) -> Dataset:
    """
    Download the big-reasoning-traces dataset and extract first N records.
    
    Args:
        source_dataset: Source dataset name on Hugging Face
        subset: Dataset subset to use (DeepSeek, OpenThoughts, etc.)
        max_records: Maximum number of records to extract
        use_streaming: Use streaming to avoid loading entire dataset in memory
        
    Returns:
        Processed dataset with first N records
    """
    logger.info(f"Downloading dataset: {source_dataset}")
    logger.info(f"Using subset: {subset}")
    logger.info(f"Extracting first {max_records:,} records")
    logger.info(f"Streaming mode: {use_streaming}")
    
    try:
        if use_streaming:
            # Use streaming for memory-efficient processing
            logger.info("Using streaming mode for memory efficiency...")
            dataset_iter = load_dataset(
                source_dataset, 
                subset, 
                split="train", 
                streaming=True
            )
            
            # Collect first N records
            records = []
            for i, record in enumerate(dataset_iter):
                if i >= max_records:
                    break
                records.append(record)
                if (i + 1) % 10000 == 0:
                    logger.info(f"Processed {i + 1:,} records...")
            
            # Convert to Dataset
            dataset = Dataset.from_list(records)
            logger.info(f"Collected {len(dataset):,} records via streaming")
            
        else:
            # Load entire dataset (uses more memory)
            logger.info("Loading entire dataset into memory...")
            dataset = load_dataset(source_dataset, subset, split="train")
            logger.info(f"Original dataset size: {len(dataset):,} records")
            
            # Extract first N records
            if len(dataset) > max_records:
                dataset = dataset.select(range(max_records))
                logger.info(f"Truncated to {len(dataset):,} records")
            else:
                logger.warning(f"Dataset has only {len(dataset):,} records, less than requested {max_records:,}")
        
        # Show dataset structure
        logger.info(f"Dataset features: {dataset.features}")
        logger.info(f"Sample record keys: {list(dataset[0].keys())}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def upload_to_huggingface(
    dataset: Dataset,
    username: str,
    dataset_name: str,
    private: bool = False
) -> str:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset: Dataset to upload
        username: Hugging Face username
        dataset_name: Name for the new dataset
        private: Whether to make the dataset private
        
    Returns:
        URL of the uploaded dataset
    """
    repo_id = f"{username}/{dataset_name}"
    
    logger.info(f"Uploading dataset to: {repo_id}")
    logger.info(f"Dataset size: {len(dataset):,} records")
    logger.info(f"Private: {private}")
    
    try:
        # Push to hub
        dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            commit_message="Initial upload: First 100k records from big-reasoning-traces"
        )
        
        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Successfully uploaded dataset to: {dataset_url}")
        
        return dataset_url
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise


def main():
    """Main function to orchestrate the download and upload process."""
    parser = argparse.ArgumentParser(
        description="Download big-reasoning-traces dataset and upload subset to Hugging Face"
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name for your new dataset"
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=100000,
        help="Maximum number of records to extract (default: 100000)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="DeepSeek",
        choices=["DeepSeek", "OpenThoughts", "OpenR1-Math"],
        help="Dataset subset to use (default: DeepSeek)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded dataset private"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for memory-efficient processing of large datasets"
    )
    
    args = parser.parse_args()
    
    # Login to Hugging Face
    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        logger.error("Hugging Face token required. Set HF_TOKEN environment variable or use --hf_token")
        return 1
    
    try:
        login(token=token)
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Failed to login to Hugging Face: {e}")
        return 1
    
    try:
        # Download and process dataset
        dataset = download_and_process_dataset(
            subset=args.subset,
            max_records=args.max_records,
            use_streaming=args.streaming
        )
        
        # Upload to Hugging Face
        dataset_url = upload_to_huggingface(
            dataset=dataset,
            username=args.hf_username,
            dataset_name=args.dataset_name,
            private=args.private
        )
        
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info(f"Dataset uploaded to: {dataset_url}")
        logger.info(f"Records: {len(dataset):,}")
        logger.info(f"Features: {list(dataset.features.keys())}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 