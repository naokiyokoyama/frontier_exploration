import argparse
import gzip
import json
import random
from pathlib import Path


def split_dataset(
    input_file: str, num_files: int, split: str, objectnav: bool = False
) -> None:
    """
    Split a PointNav or ObjectNav dataset into multiple files with balanced episode counts.

    This function performs the following tasks:
    1. Creates necessary directories
    2. Creates an empty dataset file
    3. Loads the input dataset
    4. Splits the dataset into specified number of files, ensuring equal episode counts
    5. Saves the split datasets

    Args:
        input_file (str): Path to the input json.gz file
        num_files (int): Number of files to split the dataset into
        split (str): Which split to use
        objectnav (bool): Use ObjectNav dataset instead of PointNav

    Returns:
        None
    """
    # Create necessary directories
    task = "objectnav" if objectnav else "pointnav"
    base_dir = Path(f"data/datasets/{task}/hm3d/v1/{split}")
    content_dir = base_dir / "content"
    base_dir.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(parents=True, exist_ok=True)

    # Create empty dataset file
    empty_dataset_path = base_dir / f"{split}.json.gz"
    assert empty_dataset_path.exists()

    # Load input dataset
    with gzip.open(input_file, "rt") as f:
        data = json.load(f)

    episodes = data["episodes"]
    remainder_dict = (
        {key: data[key] for key in data.keys() if key != "episodes"}
        if task == "objectnav"
        else {}
    )
    total_episodes = len(episodes)
    episodes_per_file = (total_episodes + num_files - 1) // num_files  # Round up

    # For any .json.gz files that may already be in content_dir, rename them
    for file in content_dir.iterdir():
        if file.suffix == ".gz":
            file.rename(file.with_suffix(".gz.old"))

    # Split and save datasets
    random.shuffle(episodes)  # Shuffle episodes to ensure random distribution
    for i in range(num_files):
        start = i * episodes_per_file
        end = min((i + 1) * episodes_per_file, total_episodes)
        subset = episodes[start:end]

        # If this is the last file and it's short on episodes, add random episodes
        if i == num_files - 1 and len(subset) < episodes_per_file:
            additional_needed = episodes_per_file - len(subset)
            additional_episodes = random.sample(episodes[:start], additional_needed)
            subset.extend(additional_episodes)

        stem = Path(input_file).stem[: -len(".json")]
        output_filename = f"{stem}_{i}.json.gz"
        output_path = content_dir / output_filename

        with gzip.open(output_path, "wt") as f:
            json.dump({"episodes": subset, **remainder_dict}, f)

    print(
        f"Split dataset into {num_files} files, each containing {episodes_per_file} episodes."
    )


def main() -> None:
    """
    Main function to parse arguments and call the split_dataset function.
    """
    parser = argparse.ArgumentParser(
        description="Split .json.gz dataset into multiple files with balanced episode counts."
    )
    parser.add_argument("input_file", type=str, help="Path to the input json.gz file")
    parser.add_argument("split", type=str, help="Which split to use")
    parser.add_argument(
        "num_files", type=int, help="Number of files to split the dataset into"
    )
    parser.add_argument(
        "-o",
        "--objectnav",
        action="store_true",
        help="Use ObjectNav dataset instead of PointNav",
    )
    args = parser.parse_args()

    if args.num_files <= 0:
        raise ValueError("num_files must be a positive integer")

    split_dataset(args.input_file, args.num_files, args.split, args.objectnav)


if __name__ == "__main__":
    main()
