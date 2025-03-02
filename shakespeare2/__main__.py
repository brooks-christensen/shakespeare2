import argparse
import sys

import shakespeare2.logging.logging
from shakespeare2.helpers.train import train
from shakespeare2.helpers.generator import generate


def main():

    parser = argparse.ArgumentParser(
        description="Run the generative AI model in training or generation mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "generate"],
        required=True,
        help="Mode to run: 'train' for training or 'generate' for text generation."
    )
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training mode...")
        train()  # Assumes helpers/train.py defines a main() function
    elif args.mode == "generate":
        print("Starting generation mode...")
        generate()  # Assumes helpers/generator.py defines a main() function
    else:
        print("Invalid mode. Please choose either 'train' or 'generate'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
