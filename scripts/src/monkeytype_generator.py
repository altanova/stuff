# Generate mixed character combinations from a text file of special characters
# so the final output can be used as a typing set for training with Monkeytype.
# The goal is to create a broad collection of character patterns to practice
# typing speed and accuracy with the keyboard.

import logging
import os
import random

def perfect_soup(chars: list[str], group: list[str]) -> list[str]:
    soup = []
    for g in group:
        for c in chars:
            soup.append(g)
            soup.append(c + g)
            soup.append(g + c)
    return soup

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # the input file contains groups of special characters.
    # the first line is single characters
    # the subsequent lines are pairs and triples of characters that are often found together.
    input_file = os.path.join(base_dir, 'data', 'special_characters.txt')

    # the output will contain combinations of this groups, weighted: the original groups will be repeated
    output_file = os.path.join(base_dir, 'data', 'mix.txt')
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    groups = []
    for line in lines:
        words = line.split()
        if words:
            groups.append(words)
    for g in groups:
        logging.info('group ->: %s', g)
    chars = groups[0]
    logging.info('special chars ->: %s', chars)
    # now mixing groups with single chars

    bigmix = []
    for g in groups:
        # group 0 also mixed with itself; intended
        p = perfect_soup(chars, g) 
        bigmix.extend(p)
        logging.info('expanded %d -> %d, now total %d', len(g), len(p), len(bigmix))

    random.shuffle(bigmix)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(bigmix))

if __name__ == "__main__":
    main()