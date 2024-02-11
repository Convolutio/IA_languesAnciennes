from torch import float32, zeros
from torch.testing import assert_close
from torchtext.vocab import build_vocab_from_iterator
from functools import reduce

from ..data.vocab import (computeInferenceData_Cognates,
                          computeInferenceData_Samples, wordsToOneHots)
from ..models.types import EOS_TOKEN, PADDING_TOKEN, SOS_TOKEN
from ..Source.reconstructionModel import ReconstructionModel

def testProbabilitiesSelection(debug_targets: bool = False):
    print("Test EditModel's output probabilities selection", end = "")

    LSTM_INPUT_DIM, LSTM_HIDDEN_DIM, = 100, 100
    vocabulary = build_vocab_from_iterator(
        list("abcde"), specials=[EOS_TOKEN, PADDING_TOKEN, SOS_TOKEN],
        special_first=False)
    reconModel = ReconstructionModel(
        ['french'], vocabulary, LSTM_INPUT_DIM, LSTM_HIDDEN_DIM, "cpu")
    editModel = reconModel.getModel("french")

    raw_sources = [["babedec", "dabeda"], ["cade", "abe"]]
    raw_cognates = ["cabeda", "adeb"]
    sources = computeInferenceData_Samples(
        wordsToOneHots(reduce((lambda acc, l: acc+l), raw_sources, []),
                       "cpu", vocabulary).view((7, 2, 2)),
        vocabulary)
    sources_embeddings = (reconModel.shared_embedding_layer((
        sources[0].flatten(start_dim=1), sources[1].flatten(), False
    )), sources[1], sources[2])
    fr = computeInferenceData_Cognates(
        {"french": wordsToOneHots(raw_cognates, "cpu", vocabulary)},
        vocabulary
    )["french"]

    results = editModel(sources_embeddings, fr)
    # assert the edit model's inference is referentially transparent
    for i, t in enumerate(editModel(sources_embeddings, fr)):
        assert_close(t, results[i])

    # assert the expected size is got
    assert (results[0].size() == (8, 7, 2, 2, 6)
            ), f"{results[0].size()} does not match the expected shape (8,7,1,2,6)"


    # this expected tensor is defined by direclty applying the probabilities' definitions in the paper 
    # assuming it as correct
    expected_tensors = {op: zeros((9, 8, 2, 2), dtype=float32, device="cpu")
                        for op in ("sub",
                                   "ins",
                                   "dlt", "end")}
    for c, cognate in enumerate(raw_cognates):
        source_words = raw_sources[c]
        for b, source_word in enumerate(source_words):
            if debug_targets:
                print(f"{source_word} -> {cognate}")
            expected_tensors["dlt"][:len(source_word)+1, :len(cognate)+1, c, b] = results[0][
                    :len(source_word)+1, :len(cognate)+1, c, b, -1]
            expected_tensors["end"][:len(source_word)+1, :len(cognate)+1, c, b] = results[1][
                    :len(source_word)+1, :len(cognate)+1, c, b, -1]
            for i in range(len(source_word)+1):
                for j in range(len(cognate)):
                    idx_in_dist = ord(cognate[j]) - ord('a')
                    for op_idx, op in enumerate(("sub", "ins")):
                        expected_tensors[op][i, j, c, b] = results[op_idx][i, j, c, b, idx_in_dist]
            if debug_targets:
                for op in expected_tensors:
                    print(op)
                    print(expected_tensors[op][:,:,c,b])
                print()

    # assert that the neutralization with the padding is correct
    got_results = editModel.forward_and_select(sources_embeddings, fr, True)
    for op in ("sub", "ins", "dlt", "end"):
        assert_close(got_results[op], expected_tensors[op])

    # TODO: assert that the probs are correct without padding neutralization
    # NB: requires to check on restricted zones
    print("\tOK")
