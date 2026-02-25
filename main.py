"""
CLI entry point for the YouTube Thumbnail Generator Agent.

Usage:
    venv/bin/python main.py "10 Python tips every developer must know"
    venv/bin/python main.py --prompt "How to build a SaaS in 30 days"
"""

import sys
import argparse
from agent import generate_thumbnail


def main():
    parser = argparse.ArgumentParser(
        description="Generate a YouTube thumbnail using AI (Claude + DALL-E 3)"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Description of your video / desired thumbnail",
    )
    parser.add_argument(
        "--prompt", "-p",
        dest="prompt_flag",
        help="Alternative way to pass the prompt",
    )
    args = parser.parse_args()

    prompt = args.prompt or args.prompt_flag
    if not prompt:
        prompt = input("Enter your video topic or thumbnail description:\n> ").strip()
        if not prompt:
            print("Error: prompt cannot be empty.")
            sys.exit(1)

    print(f"\nGenerating thumbnail for: \"{prompt}\"\n{'─' * 60}")

    result = generate_thumbnail(prompt)

    print(f"{'─' * 60}")

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)

    print(f"\n✓ Thumbnail saved to: {result['output_path']}")
    print(f"\n--- Topic Analysis ---\n{result['topic_analysis']}")
    print(f"\n--- Imagen 3 Prompt Used ---\n{result['image_prompt']}")


if __name__ == "__main__":
    main()
