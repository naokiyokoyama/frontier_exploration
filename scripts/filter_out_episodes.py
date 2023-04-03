import argparse
import glob
import gzip
import hashlib
import json
import os
import os.path as osp
import shutil

import tqdm

HASH_KEYS = ["scene_id", "start_position", "start_rotation"]


def main(episodes_to_remove_paths, input_dataset, output_dataset):
    episodes_to_remove = []
    print("Reading directories...")
    for remove_dir in tqdm.tqdm(episodes_to_remove_paths):
        list_file = osp.join(remove_dir, osp.basename(remove_dir) + ".list")
        if osp.exists(list_file):
            with open(list_file, "r") as f:
                episode_hashes = f.read().splitlines()
        else:
            txt_files = glob.glob(osp.join(remove_dir, "*.txt"))
            episode_hashes = [osp.basename(j)[:-4] for j in txt_files]
            with open(list_file, "w") as f:
                f.write("\n".join(episode_hashes))
        episodes_to_remove.extend(episode_hashes)
    remove_episode_hashes = set(episodes_to_remove)
    print(f"Found {len(remove_episode_hashes)} episodes to remove")

    gz_files = glob.glob(osp.join(input_dataset, "content", "*.json.gz"))
    print(f"Found {len(gz_files)} gz_files")
    new_episode_count = 0
    full_episode_count = 0
    dataset_initialized = False
    for gz_file in tqdm.tqdm(gz_files):
        with gzip.open(gz_file, "rt") as f:
            data_dict = json.load(f)

        filtered_episodes = []
        gz_remove_count = 0
        for ep in tqdm.tqdm(data_dict["episodes"]):
            strs = [ep[k] for k in HASH_KEYS]
            strs[0] = osp.basename(strs[0])
            hash_str = ":".join([str(i) for i in strs])
            hash_str = hashlib.sha224(hash_str.encode("ASCII")).hexdigest()
            if hash_str not in remove_episode_hashes:
                filtered_episodes.append(ep)
                new_episode_count += 1
            else:
                gz_remove_count += 1
            full_episode_count += 1
        print(f"\nRemoved {gz_remove_count} episodes from {osp.basename(gz_file)}")

        if len(filtered_episodes) > 0:
            if not dataset_initialized:
                setup_dataset(input_dataset, output_dataset)
                dataset_initialized = True
            filtered_data_dict = data_dict.copy()
            filtered_data_dict["episodes"] = filtered_episodes
            output_gz = osp.join(output_dataset, "content", osp.basename(gz_file))
            with gzip.open(output_gz, "wt") as f:
                json.dump(filtered_data_dict, f)

    print(f"Original episode count: {full_episode_count}")
    print(f"New episode count: {new_episode_count}")
    removed = full_episode_count - new_episode_count
    print(
        f"Removed {removed} out of {len(episodes_to_remove)} "
        f"({removed / len(episodes_to_remove) * 100}%) episodes"
    )
    print(f"New dataset is at: {output_dataset}")


def setup_dataset(input_dataset, output_dataset):
    os.makedirs(osp.join(output_dataset, "content"), exist_ok=True)
    base_gz = glob.glob(osp.join(input_dataset, "*.json.gz"))
    assert len(base_gz) == 1
    shutil.copyfile(
        base_gz[0], osp.join(output_dataset, osp.basename(output_dataset) + ".json.gz")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "episodes_to_remove",
        type=str,
        help="Path(s) to the episode directory(s). Comma separated if multiple.",
    )
    parser.add_argument(
        "input_dataset",
        type=str,
        help="Path to the full episodes directory (containing 'content/*json.gz')",
    )
    parser.add_argument(
        "output_dataset",
        type=str,
        help="Path to new episodes directory",
    )
    args = parser.parse_args()

    args.episodes_to_remove = args.episodes_to_remove.split(",")
    for i in args.episodes_to_remove:
        if not osp.exists(i):
            raise ValueError(f"Episode directory {i} does not exist")
    if not osp.exists(args.input_dataset):
        raise ValueError(f"Input directory {args.input_dataset} does not exist")
    if osp.exists(args.output_dataset):
        raise ValueError(f"Output directory {args.output_dataset} already exists")

    main(args.episodes_to_remove, args.input_dataset, args.output_dataset)
