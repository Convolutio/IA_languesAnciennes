import torch
from data.vocab import vocabulary
from models.types import PADDING_TOKEN

MAX_WORD_LENGTH = 20

# batch_sizes = [randint(5, MAX_SAMPLES_NUMBER) for _ in range(C)]
def createSamplesBatch(cognates_groups_number: int, samples_number_per_cognates_group: int) -> list[torch.Tensor]:
    lengths = [torch.randint(low = 3, high = MAX_WORD_LENGTH, size = (samples_number_per_cognates_group,)) for _ in range(cognates_groups_number)]
    batch = []
    for i in range(cognates_groups_number):
        sequenceLengths = lengths[i]
        maxLength = int(torch.max(sequenceLengths).item())
        notPaddingTokenPositions = torch.arange(maxLength).unsqueeze(1) < sequenceLengths.unsqueeze(0)
        samplesTensor = torch.randint(0, len(vocabulary)-3,
                                        (maxLength, samples_number_per_cognates_group))
        batch.append(samplesTensor.where(notPaddingTokenPositions, vocabulary[PADDING_TOKEN]).T)
    return batch