import argparse
import glob
import gzip
import json
import os
import os.path as osp
import shutil

import tqdm

HASH_KEYS = ["episode_id", "scene_id", "start_position", "start_rotation"]


def main(multistory_episodes, input_dataset, output_dataset):
    multistory_episodes = glob.glob(osp.join(multistory_episodes, "*.txt"))
    print(f"Found {len(multistory_episodes)} multistory episodes")
    multistory_episode_hashes = set()
    for episode in multistory_episodes:
        with open(episode, "r") as f:
            multistory_episode_hashes.add(refine_hash(f.read()))

    os.makedirs(osp.join(output_dataset, "content"), exist_ok=True)
    base_gz = glob.glob(osp.join(input_dataset, "*.json.gz"))
    assert len(base_gz) == 1
    shutil.copyfile(base_gz[0], osp.join(output_dataset, osp.basename(base_gz[0])))

    gz_files = glob.glob(osp.join(input_dataset, "content", "*.json.gz"))
    print(f"Found {len(gz_files)} gz_files")
    new_episode_count = 0
    full_episode_count = 0
    for gz_file in tqdm.tqdm(gz_files):
        with gzip.open(gz_file, "rt") as f:
            data_dict = json.load(f)

        filtered_episodes = []
        for ep in data_dict["episodes"]:
            strs = [ep[k] for k in HASH_KEYS]
            raw_hash = ":".join([str(i) for i in strs])
            new_hash = refine_hash(raw_hash)
            if new_hash not in multistory_episode_hashes:
                filtered_episodes.append(ep)
                new_episode_count += 1
            full_episode_count += 1

        filtered_data_dict = data_dict.copy()
        filtered_data_dict["episodes"] = filtered_episodes
        output_gz = osp.join(output_dataset, "content", osp.basename(gz_file))
        with gzip.open(output_gz, "wt") as f:
            json.dump(filtered_data_dict, f)

    print(f"New episode count: {new_episode_count}")
    print(f"Full episode count: {full_episode_count}")
    removed = full_episode_count - new_episode_count
    print(
        f"Removed {removed} out of {len(multistory_episodes)} "
        f"({removed / len(multistory_episodes) * 100}%) episodes"
    )


def refine_hash(raw_hash):
    new_hash = raw_hash.split(":")
    new_hash[1] = osp.basename(new_hash[1])
    new_hash.pop(0)
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

    if not osp.exists(args.multistory_episodes):
        raise ValueError(
            f"Multistory episodes directory {args.multistory_episodes} does not exist"
        )
    if not osp.exists(args.input_dataset):
        raise ValueError(f"Input directory {args.input_dataset} does not exist")
    if osp.exists(args.output_dataset):
        raise ValueError(f"Output directory {args.output_dataset} already exists")

    main(args.multistory_episodes, args.input_dataset, args.output_dataset)
