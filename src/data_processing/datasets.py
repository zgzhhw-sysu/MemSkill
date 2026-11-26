"""Load supported datasets and apply the MemSkill evaluation splits."""
import json
import os


def load_dataset(data_file: str, dataset_type: str):
    """Load dataset"""
    with open(data_file, 'r') as f:
        if dataset_type == 'locomo':
            data = json.load(f)
        elif dataset_type == 'longmemeval':
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f.readlines()]
        elif dataset_type == 'hotpotqa':
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f.readlines()]
        elif dataset_type == 'alfworld':
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f.readlines()]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    return data


def split_data(data, dataset_type: str):
    """Split data into train/val/test"""
    if dataset_type == 'locomo':
        # LoCoMo uses fixed indices
        train_index = [0, 1, 2, 3, 4, 5]
        val_index = [6, 7]
        test_index = [8, 9]

        train_data = [data[i] for i in train_index if i < len(data)]
        val_data = [data[i] for i in val_index if i < len(data)]
        test_data = [data[i] for i in test_index if i < len(data)]

    elif dataset_type == 'longmemeval':
        # LongMemEval uses predefined splits
        splits_file = "./data/longmemeval_s_splits.json"
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            train_idx = splits["train"]
            val_idx = splits["val"]
            test_idx = splits["test"]

            train_data = [data[i] for i in train_idx if i < len(data)]
            val_data = [data[i] for i in val_idx if i < len(data)]
            test_data = [data[i] for i in test_idx if i < len(data)]
        else:
            raise FileNotFoundError

    elif dataset_type == 'hotpotqa':
        train_data = data
        val_data = []
        test_data = []
    elif dataset_type == 'alfworld':
        if isinstance(data, dict):
            train_data = data
            val_data = {}
            test_data = {}
        else:
            n = len(data)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return train_data, val_data, test_data
