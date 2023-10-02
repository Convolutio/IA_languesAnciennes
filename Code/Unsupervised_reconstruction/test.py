import time
from time import time

# from lm.PriorLM import NGramLM
# from data.ipa_tokenizer import tokenize_ipa
# from data.vocab import vocabulary

def functionToRun():
    """
    Call here the function
    """
    from data import datapipes
    # bigram = NGramLM(3, vocabulary)
    # txt = ""
    # with open('latin_text_ipa.txt', 'r', encoding='utf-8') as f:
    #     txt = tokenize_ipa(f.read().split(" "), "".join(vocabulary.get_itos()), 20000, None)
    # bigram.train(txt)

if __name__=="__main__":
    start_time = time()
    functionToRun()
    duration = time() - start_time #in seconds

    print(f'\n\nExecution time : {duration//60} minutes and {duration%60} seconds')