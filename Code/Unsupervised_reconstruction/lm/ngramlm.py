import numpy as np
from collections import Counter
from nltk.util import ngrams
from ipa_tokenizer import tokenize_ipa

class NgramLM:
    """
    Ngram Language Model.
    """

    def __init__(self, n: int):
        self.n: int = n
        self.vocab: list[str]
        self.ngram_counts: Counter = Counter()
        self.ngram_minus_counts: Counter = Counter()
        self.ngrams_distrib: dict

    def get_distrib(self):
        """Returns the dictionnary of each ngram with its MLE, learned by the language model."""
        return self.ngrams_distrib

    def train(self, tokens: list[str]):
        """
        Train the language model based on a list of tokens.

        Arg:
            - tokens (list[str]): Tokenized training set.
        
        Returns:
            - dict: For each ngram key correspond to its MLE.
        """

        # TODO: Kneser–Ney smoothing ?

        # Split tokens into an iterable of ngrams, then count each occurrence of ngrams, stocked into a Counter.
        self.ngram_counts.update(Counter(ngrams(tokens, self.n)))
        self.ngram_minus_counts.update(Counter(ngrams(tokens, self.n-1)))

        # Speech and Language Processing by Jurasky, Chap 3, eq 3.12
        self.ngrams_distrib = {ngram: self.ngram_counts[ngram] / self.ngram_minus_counts[ngram[:-1]] for ngram in self.ngram_counts}

        return self.ngrams_distrib
    
    def score_sequence(self, tokens: list[str]) -> float:
        """
        Give the probability of a tokenised sequence.

        Arg:
            - tokens (list[str]): Tokenised input sequence.

        Returns:
            - float: The probability of the sequence.
        """

        ngram_tokens = ngrams(tokens, self.n)
        log_prob = 0.0

        for ngram in ngram_tokens:
            # Speech and Language Processing by Jurasky, Chap 3, eq 3.4
            log_prob+=np.log(self.ngrams_distrib[ngram])        # TODO: Handle ngram never seen
        
        return np.exp(log_prob)


"""
Test.
"""

if __name__ == '__main__':

    """ Set the IPA vocabulary """
    # TODO: Get directly the voc on Github
    vocab_IPA = "zɣmɒusʲˈːʔɨokpeaβøbfɡʒyɲɾˌɛdɹwxnlrœɐʁvʌʊŋʝʰʎjhðʃɪ\-ɔəɑiθt\u0303"

    """ Open the latin text in IPA """
    # TODO: Create a new file latin_text_ipa cause it missing voc chr in the text (then re-check the voc)
    with open('latin_text_ipa.txt', 'r', encoding='utf-8') as f:
        ipa_text_one = f.read() 

    with open('latin_ipa_default.txt', 'r', encoding='utf-8') as f:
        ipa_text_two = f.read() 

    """ Tokenize the IPA text """
    tokens_one = tokenize_ipa(ipa_text_one, vocab_IPA)
    tokens_two = tokenize_ipa(ipa_text_two, vocab_IPA)

    """ Generate and train ngram language model """
    # TODO: Create a loop to build ngram language models for different size of text (7 000 > 12 000 > 25 000)
    # Training on 5k words of default word + custom words (2k at 20k) 
    bigram = NgramLM(2)
    bigram.train(tokens_one)
    bigram.train(tokens_two)

    trigram = NgramLM(3)
    trigram.train(tokens_one)
    
    """ Evaluation """
    sentence_tokenize = tokenize_ipa("arɡymntaθjon", vocab_IPA)
    print(bigram.score_sequence(sentence_tokenize))
    print(trigram.score_sequence(sentence_tokenize))
