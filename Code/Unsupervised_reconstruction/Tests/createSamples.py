from torch import Tensor
import torch
from random import randint
from data.vocab import vocabulary
from models.models import PADDING_TOKEN

C = 10
MAX_SAMPLES_NUMBER = 30
MAX_WORD_LENGTH = 16

# batch_sizes = [randint(5, MAX_SAMPLES_NUMBER) for _ in range(C)]
batch_sizes = [MAX_SAMPLES_NUMBER for _ in range(C)]
lengths = [torch.randint(low = 3, high = MAX_WORD_LENGTH, size = (batch_sizes[i],)) for i in range(C)]
batch = []
for i in range(C):
    sequenceLengths = lengths[i]
    maxLength = int(torch.max(sequenceLengths).item())
    notPaddingTokenPositions = torch.arange(maxLength).unsqueeze(1) < sequenceLengths.unsqueeze(0)
    samplesTensor = torch.randint(0, len(vocabulary)-3,
                                       (maxLength, batch_sizes[i]))
    batch.append(samplesTensor.where(notPaddingTokenPositions, vocabulary[PADDING_TOKEN]))
    
torch.save(batch, './tests/samples.pt')