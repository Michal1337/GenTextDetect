import os
from typing import List


def get_csv_paths(folder_path: str, recursive: bool = False) -> List[str]:
    if recursive:
        # Walk through all subdirectories
        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(folder_path)
            for file in files
            if file.endswith(".csv")
        ]
    else:
        # Get files in the root folder only
        file_paths = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if file.endswith(".csv")
        ]

    return file_paths
