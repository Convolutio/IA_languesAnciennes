import re
import itertools
from nltk.tokenize import word_tokenize

def tokenize_ipa(ipa_string: str, ipa_vocab: str, save_txt: bool = False) -> list[str]:
    """
    Custom tokenizer to process only words with the determined IPA vocabulary.
    
    Args:
        - ipa_string (str): The input text.
        - ipa_vocab (str): String of all element in vocabulary ()
        - save_txt (bool): Option to save the tokenized text or not. Defaults to `False`.

    Returns:
        - list[str]: The tokenized text in a Python list.
    """

    # TODO: Correct the voc + Use directly the vocab on github
    # ipa_pattern = r"\b[{0}]+\b".format(ipa_vocab)
    ipa_regex = re.compile(r"^[{0}]+$".format(ipa_vocab))

    tokens = []

    # Keep only words that are written with the defined vocabulary.
    for word in word_tokenize(ipa_string):
        if ipa_regex.match(word):
            tokens.append(word)

    # TODO: Handle diacrits (or similar characters)
    tokens = list(itertools.chain.from_iterable(tokens))

    # Save the tokenized text into a plain text file.
    if save_txt:
        with open("tokenized_ipa.txt", "w", encoding='utf-8') as f:
            f.write(" ".join(tokens))
    
    return tokens