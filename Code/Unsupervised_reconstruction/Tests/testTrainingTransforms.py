from torchdata.datapipes.map import SequenceWrapper
from torchdata.dataloader2 import DataLoader2
from data.getDataset import getCognatesSet, getIteration
from data.datapipes import formatTargets, get_training_datapipe
from data.vocab import computeInferenceData, wordsToOneHots, vocabulary
from models.articleModels import ModernLanguages, MODERN_LANGUAGES, Operations
from models.models import InferenceData
from source.reconstructionModel import ReconstructionModel

raw_cognates = getCognatesSet()
cognates:dict[ModernLanguages, InferenceData] = formatTargets(raw_cognates)
raw_samples = getIteration(1)
currentReconstructions = computeInferenceData(wordsToOneHots(raw_samples).unsqueeze(-1)) #TODO: simplify the data loading

LSTM_INPUT_DIM = 50
LSTM_HIDDEN_DIM = 50

randomEditModel = ReconstructionModel(MODERN_LANGUAGES, vocabulary, LSTM_INPUT_DIM, LSTM_HIDDEN_DIM)

TEST_LANGUAGE:ModernLanguages = "french"
x_maxLength = currentReconstructions[0].size()[0] - 2
y_maxLength = cognates[TEST_LANGUAGE][0].size()[0] - 2
print('|y| max =', y_maxLength)
print('|x| max =', x_maxLength)

target_probs = randomEditModel.backward_dynProg(currentReconstructions, cognates) #type: ignore

for lang in target_probs.keys(): #type: ignore
    target_probs[lang] = target_probs[lang].toTargetsProbs() #type: ignore

# list (len=C) of dict[ModernLanguages, dict[Operations, Tensor(shape=*)]]
target_probs: list[dict[ModernLanguages, dict[Operations, Tensor]]] = [dict(zip(target_probs,t)) for t in zip(*target_probs.values())] #type: ignore

print(len(target_probs))
print(target_probs[156].keys())
print(target_probs[148]['french'].keys())
print(target_probs[89]['portuguese']['dlt'].size())
dp = SequenceWrapper(target_probs)
dl = DataLoader2(dp) # TODO: remove this memory leak

#If the code above works, then the ideal code below should works
MINI_BATCH_SIZE = 30
dp2 = SequenceWrapper(raw_samples).zip(SequenceWrapper(raw_cognates), SequenceWrapper(target_probs))
dp2 = get_training_datapipe(dp2, MINI_BATCH_SIZE) #type: ignore
dl2 = DataLoader2(dp2)