import argparse
import json
import os

import cv2
import numpy as np


def load_and_compare_frontiers(json_path, img_dir):
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract timestep_to_frontiers
    timestep_to_frontiers = data["timestep_to_frontiers"]

    # Sort timesteps
    sorted_timesteps = sorted(timestep_to_frontiers.keys(), key=int)

    for timestep in sorted_timesteps:
        frontier_ids = timestep_to_frontiers[timestep]["frontier_ids"]

        # Only process sets with more than one frontier
        if len(frontier_ids) > 1:
            images = []
            for frontier_id in frontier_ids:
                img_path = os.path.join(img_dir, f"frontier_{frontier_id:04d}.jpg")
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                images.append(img)

            if images:
                # Ensure all images have the same height
                min_height = min(img.shape[0] for img in images)
                resized_images = [
                    cv2.resize(
                        img, (int(img.shape[1] * min_height / img.shape[0]), min_height)
                    )
                    for img in images
                ]

                # Horizontally stack images
                combined_image = np.hstack(resized_images)

                # Display combined image
                cv2.imshow(f"Frontiers at timestep {timestep}", combined_image)
                print(f"Showing frontiers {frontier_ids} for timestep {timestep}")
                print("Press any key to continue to the next set, or 'q' to quit")

                # Wait for key press
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()

                if key == ord("q"):
                    break


def main():
    parser = argparse.ArgumentParser(
        description="Compare frontier images based on JSON data."
    )
    parser.add_argument("json_path", type=str, help="Path to the JSON file")
    parser.add_argument(
        "img_dir", type=str, help="Path to the directory containing frontier images"
    )

    args = parser.parse_args()

    load_and_compare_frontiers(args.json_path, args.img_dir)


if __name__ == "__main__":
    main()
