import re
import random
from typing import Optional

# Fix the randomness
random.seed(42)


def tokenize_ipa(data: list[str], ipa_vocab: str, sample, filename: Optional[str]) -> str:
    """
    Custom tokenizer to process only words with the determined IPA vocabulary.

    Args:
        ipa_string (list[str]): The input text.
        ipa_vocab (str): String of all element in vocabulary
        filename (Optional[str]): name of the output file.

    Returns:
        str: The tokenized text in a string.
    """

    ipa_regex = re.compile(r"^[{0}]+$".format(ipa_vocab))

    tokens = []

    for word in data:
        if ipa_regex.match(word):
            tokens.append(word)

    tokens = random.sample(tokens, sample)

    if filename:
        with open(f"{filename}.txt", "w", encoding='utf-8') as f:
            f.write(" ".join(tokens))

    return " ".join(tokens)
