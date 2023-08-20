from data.ipa_tokenizer import tokenize_ipa


def gen_train_data() -> tuple[str, ...]:
    voc = ""
    raw_data = ""
    max_words = 20_000  # cf 'Article Scientifique'

    data = []

    with open('./data/IPA_vocabulary.txt', 'r', encoding='utf-8') as vocFile:
        voc = vocFile.read().replace(', ', "")
        print(voc)

    # Dataset source : https://huggingface.co/datasets/pstroe/cc100-latin
    with open('./latin_text_ipa.txt', 'r', encoding='utf-8') as dataFile:
        raw_data = dataFile.read()
        print(raw_data[:100])

    for i in range(0, 3):
        nb_word_samples = max_words - 5000 * i
        data.append(tokenize_ipa(raw_data.split(' '), voc,
                     nb_word_samples, filename=f"{nb_word_samples}_latin_tokens"))

        with open(f'./{nb_word_samples}_latin_tokens.txt', 'r', encoding='utf-8') as f:
            print(f.read()[:100])

    return tuple(data)
