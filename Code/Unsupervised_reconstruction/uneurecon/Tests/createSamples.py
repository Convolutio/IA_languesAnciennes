import torch
from torch.types import Device
from torchtext.vocab import Vocab

from ..models.types import PADDING_TOKEN

MAX_WORD_LENGTH = 20


def createSamplesBatch(cognates_groups_number: int, samples_number_per_cognates_group: int, device:
                       Device, vocabulary: Vocab) -> list[torch.Tensor]:
    recons_lengths = torch.randint(
        low=3, high=MAX_WORD_LENGTH+1, size=(cognates_groups_number,))
    lengths = [torch.randint(low=int(recons_lengths[i].item()-2), high=int(recons_lengths[i]
                                                                           .item() + 3),
                             size=(samples_number_per_cognates_group,))
               for i in range(cognates_groups_number)]
    batch = []
    for i in range(cognates_groups_number):
        sequenceLengths = lengths[i]
        maxLength = int(torch.max(sequenceLengths).item())
        notPaddingTokenPositions = torch.arange(
            maxLength).unsqueeze(1) < sequenceLengths.unsqueeze(0)
        samplesTensor = torch.randint(low=0, high=len(vocabulary)-3,
                                      size=(
                                          maxLength, samples_number_per_cognates_group),
                                      dtype=torch.uint8)
        samplesTensor = samplesTensor.where(
            notPaddingTokenPositions, vocabulary[PADDING_TOKEN]).T
        batch.append(samplesTensor.to(device))
    return batch
