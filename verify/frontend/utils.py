from verify.backend.datasets.loader import count_dataset_items as _backend_count_dataset_items

def count_dataset_items(dataset_name: str, modality: str) -> int:
    """Return the total number of items in a dataset."""
    try:
        return _backend_count_dataset_items(dataset_name, modality)
    except Exception:
        return 0
