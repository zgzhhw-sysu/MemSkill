"""Command-line training and evaluation for MemSkill."""
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

from src.config import AgenticMemoryConfig, get_agentic_memory_args
from src.data_processing.datasets import load_dataset, split_data
from src.inference.alfworld import infer_alfworld_memories
from src.inference.text import infer_text_dataset_memories
from src.trainer import get_trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Parse arguments
    args = get_agentic_memory_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = AgenticMemoryConfig()
    config.update_from_args(args)

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Load dataset
    data_file = args.data_file
    if args.dataset == "alfworld" and getattr(args, "alfworld_offline_data", None):
        data_file = args.alfworld_offline_data
    print(f"Loading dataset from {data_file}")
    data = load_dataset(data_file, args.dataset)
    eval_data = None
    if args.dataset == "hotpotqa":
        eval_file = getattr(args, "hotpotqa_eval_file", "./data/eval_200.json")
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"HotpotQA eval file not found: {eval_file}")
        print(f"Loading HotpotQA eval dataset from {eval_file}")
        eval_data = load_dataset(eval_file, args.dataset)
    if args.dataset == "alfworld":
        eval_file = getattr(args, "alfworld_eval_file", "./data/alfworld_expert_eval_in_distribution.json")
        if not os.path.exists(eval_file):
            raise FileNotFoundError(f"ALFWorld eval file not found: {eval_file}")
        print(f"Loading ALFWorld eval dataset from {eval_file}")
        eval_data = load_dataset(eval_file, args.dataset)

    # Split data
    train_data, val_data, test_data = split_data(data, args.dataset)
    if args.dataset == "hotpotqa" and eval_data is not None:
        test_data = eval_data
        val_data = []
    if args.dataset == "alfworld" and eval_data is not None:
        test_data = eval_data
        val_data = {}
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Initialize trainer
    print("\nInitializing Agentic Memory Trainer...")
    trainer = get_trainer(args, config)

    def _run_inference():
        if args.dataset == "alfworld":
            return infer_alfworld_memories(trainer, train_data, test_data, args)
        return infer_text_dataset_memories(trainer, test_data, args)

    if args.eval_only:
        # Load checkpoint and infer
        if args.load_checkpoint is None:
            raise ValueError("Must specify --load-checkpoint for inference")

        print(f"Loading checkpoint from {args.load_checkpoint}")
        trainer.load_checkpoint(args.load_checkpoint)

        # Infer and save memories
        _run_inference()

    else:
        if args.load_checkpoint is not None:
            print(f"Loading checkpoint from {args.load_checkpoint}")
            trainer.load_checkpoint(args.load_checkpoint)
        # Train
        print("\nStarting training...")
        trainer.train(train_data)

        # Save final model
        trainer.save_checkpoint('final')

        # Infer on test set and save memories
        # print("\nInferring on test set and saving memories...")
        # _run_inference()

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == "__main__":
    main()
