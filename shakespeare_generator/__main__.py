import argparse
import sys

import shakespeare_generator.logging.logging
from shakespeare_generator.helpers.train import train
from shakespeare_generator.helpers.generator import generate


def main():

    parser = argparse.ArgumentParser(
        description="Run the generative AI model in training or generation mode."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Subcommands: 'train' or 'generate'")

    # Subparser for 'train' mode
    train_parser = subparsers.add_parser("train", help="Run the model in training mode.")

    # Subparser for 'generate' mode
    generate_parser = subparsers.add_parser("generate", help="Run the model in text generation mode.")
    
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training mode...")
        train()  
    elif args.mode == "generate":
        print("Starting generation mode...")
        generate() 
    else:
        print("Invalid mode. Please choose either 'train' or 'generate'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
