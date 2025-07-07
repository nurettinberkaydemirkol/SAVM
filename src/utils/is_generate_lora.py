import os

def is_generate_lora(dataset_file: str = None) -> bool:
    """
    Check if the dataset file is a valid JSON file with more than 20 lines.

    Args:
        dataset_file (str): The path to the dataset file.

    Returns:
        bool: True if file exists and has more than 20 lines, False otherwise.
    """
    if not dataset_file or not dataset_file.strip():
        return False

    if not os.path.isfile(dataset_file):
        return False

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        return line_count > 20
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return False