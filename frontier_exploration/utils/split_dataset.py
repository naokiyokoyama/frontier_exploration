import argparse
import gzip
import json
from pathlib import Path


def split_pointnav_dataset(input_file: str, num_files: int) -> None:
    """
    Split a PointNav dataset into multiple files.

    This function performs the following tasks:
    1. Creates necessary directories
    2. Creates an empty dataset file
    3. Loads the input dataset
    4. Splits the dataset into specified number of files
    5. Saves the split datasets

    Args:
        input_file (str): Path to the input json.gz file
        num_files (int): Number of files to split the dataset into

    Returns:
        None
    """
    # Create necessary directories
    base_dir = Path("data/datasets/pointnav/hm3d/v1/val")
    content_dir = base_dir / "content"
    base_dir.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(parents=True, exist_ok=True)

    # Create empty dataset file
    empty_dataset_path = base_dir / "val.json.gz"
    with gzip.open(empty_dataset_path, "wt") as f:
        json.dump({"episodes": []}, f)

    # Load input dataset
    with gzip.open(input_file, "rt") as f:
        data = json.load(f)

    episodes = data["episodes"]
    total_episodes = len(episodes)
    episodes_per_file = total_episodes // num_files
    remainder = total_episodes % num_files

    # Split and save datasets
    start = 0
    for i in range(num_files):
        end = start + episodes_per_file + (1 if i < remainder else 0)
        subset = episodes[start:end]

        stem = Path(input_file).stem[: -len(".json")]
        output_filename = f"{stem}_{i}.json.gz"
        output_path = content_dir / output_filename

        with gzip.open(output_path, "wt") as f:
            json.dump({"episodes": subset}, f)

        start = end


def main() -> None:
    """
    Main function to parse arguments and call the split_pointnav_dataset function.
    """
    parser = argparse.ArgumentParser(
        description="Split PointNav dataset into multiple files."
    )
    parser.add_argument("input_file", type=str, help="Path to the input json.gz file")
    parser.add_argument(
        "num_files", type=int, help="Number of files to split the dataset into"
    )
    args = parser.parse_args()

    if args.num_files <= 0:
        raise ValueError("num_files must be a positive integer")

    split_pointnav_dataset(args.input_file, args.num_files)
    print(f"Split dataset into {args.num_files} files!")


if __name__ == "__main__":
    main()
