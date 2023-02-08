import argparse
import glob
import gzip
import json
import os

import numpy as np
import tqdm

HASH_KEYS = ["episode_id", "scene_id", "start_position", "start_rotation"]


def main(multistory_episodes, input_dataset, output_dataset):
    multistory_episodes = glob.glob(os.path.join(multistory_episodes, "*.txt"))
    print(f"Found {len(multistory_episodes)} multistory episodes")

    multistory_episode_hashes = set()
    for episode in multistory_episodes:
        with open(episode, "r") as f:
            multistory_episode_hashes.add(refine_hash(f.read()))

    os.makedirs(os.path.join(output_dataset, "content"), exist_ok=True)

    gz_files = glob.glob(os.path.join(input_dataset, "content", "*.json.gz"))
    print(f"Found {len(gz_files)} gz_files")
    new_episode_count = 0
    full_episode_count = 0
    for gz_file in tqdm.tqdm(gz_files):
        with gzip.open(gz_file, "rt") as f:
            data_dict = json.load(f)

        filtered_episodes = []
        for ep in data_dict["episodes"]:
            full_episode_count += 1
            strs = [ep[k] for k in HASH_KEYS]
            raw_hash = ":".join([str(i) for i in strs])
            new_hash = refine_hash(raw_hash)
            if new_hash not in multistory_episode_hashes:
                filtered_episodes.append(ep)
                new_episode_count += 1

        filtered_data_dict = data_dict.copy()
        filtered_data_dict["episodes"] = filtered_episodes
        output_gz = os.path.join(output_dataset, "content", os.path.basename(gz_file))
        with gzip.open(output_gz, "wt") as f:
            json.dump(filtered_data_dict, f)

    print(f"New episode count: {new_episode_count}")
    print(f"Full episode count: {full_episode_count}")


def refine_hash(raw_hash):
    new_hash = raw_hash.split(":")
    new_hash[1] = os.path.basename(new_hash[1])
    for i in [2, 3]:
        new_hash[i] = np.array(eval(new_hash[i]))
        new_hash[i] = str(np.round(new_hash[i], 1))
    return ":".join(new_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "multistory_episodes",
        type=str,
        help="Path to the multistory episodes directory",
    )
    parser.add_argument(
        "input_dataset",
        type=str,
        help="Path to the full episodes directory",
    )
    parser.add_argument(
        "output_dataset",
        type=str,
        help="Path to new episodes directory",
    )
    args = parser.parse_args()

    if not os.path.exists(args.multistory_episodes):
        raise ValueError(
            f"Multistory episodes directory {args.multistory_episodes} does not exist"
        )
    if not os.path.exists(args.input_dataset):
        raise ValueError(f"Input directory {args.input_dataset} does not exist")
    if os.path.exists(args.output_dataset):
        raise ValueError(f"Output directory {args.output_dataset} already exists")

    main(args.multistory_episodes, args.input_dataset, args.output_dataset)
