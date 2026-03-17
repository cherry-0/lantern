# imports
from verify.backend.datasets.loader import get_dataset_path, _is_hf_dataset, list_dataset_items
import json

def count_dataset_items(dataset_name: str, modality: str) -> int:
    """
    Return the total number of items in a dataset without loading all data.
    For HuggingFace datasets, reads num_examples from dataset_info.json.
    For flat file datasets, counts matching files.
    """
    dataset_path = get_dataset_path(dataset_name)
    if dataset_path is None:
        return 0

    if _is_hf_dataset(dataset_path):
        total = 0
        for split_dir in dataset_path.iterdir():
            if not split_dir.is_dir():
                continue
            info_file = split_dir / "dataset_info.json"
            if info_file.exists():
                try:
                    info = json.loads(info_file.read_text())
                    for split_info in info.get("splits", {}).values():
                        total += split_info.get("num_examples", 0)
                except Exception:
                    pass
        return total

    return len(list_dataset_items(dataset_name, modality))