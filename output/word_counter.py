# word_counter.py

import argparse
import os


def count_words_and_characters(text):
    words = len(text.split())
    characters = len(text)
    return words, characters


def main():
    parser = argparse.ArgumentParser(
        description="Count words and characters from input text or file."
    )
    parser.add_argument("-f", "--file", type=str, help="File path to read text from.")
    args = parser.parse_args()

    text = ""
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: The file {args.file} does not exist.")
            return
        with open(args.file, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        print("Enter text (Ctrl-D to end):")
        text += "".join(iter(input, ""))

    words, characters = count_words_and_characters(text)
    print(f"Words: {words}")
    print(f"Characters: {characters}")


if __name__ == "__main__":
    main()
