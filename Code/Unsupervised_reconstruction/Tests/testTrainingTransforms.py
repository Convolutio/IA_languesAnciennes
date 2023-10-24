from torch.utils.data import Dataset, default_collate
# from torchdata.dataloader2 import DataLoader2
from torch.utils.data import DataLoader
from torch import Tensor
from data.getDataset import getCognatesSet, getIteration
from data.datapipes import formatTargets
from data.vocab import computeInferenceData_Samples, wordsToOneHots, vocabulary
from models.types import ( ModernLanguages, MODERN_LANGUAGES, Operations, OPERATIONS, InferenceData, PADDING_TOKEN )
from source.reconstructionModel import ReconstructionModel
from source.utils import pad2d_sequence

raw_cognates = getCognatesSet()
cognates:dict[ModernLanguages, InferenceData] = formatTargets(raw_cognates)
raw_samples = getIteration(1)
currentReconstructions = computeInferenceData_Samples(wordsToOneHots(raw_samples).unsqueeze(-1)) #TODO: simplify the data loading

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
CachedTargetProbs = dict[ModernLanguages, dict[Operations, Tensor]] 
target_probs: list[CachedTargetProbs] = [dict(zip(target_probs,t)) for t in zip(*target_probs.values())] #type: ignore

print(len(target_probs))
print(target_probs[156].keys())
print(target_probs[148]['french'].keys())
print(target_probs[89]['portuguese']['dlt'].size())

class MyDataset(Dataset):
    def __init__(self, raw_samples, raw_cognates, target_probs: list[dict[ModernLanguages, dict[Operations, Tensor]]]) -> None:
        self.training_load = list(zip(raw_samples, raw_cognates, target_probs))
        self.length = len(target_probs)

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index):
        return self.training_load[index]
    
def __training__collate_fn(batch:list[tuple[str, dict[ModernLanguages, str], CachedTargetProbs]]):
    """
    Collates the input and target data in the batch.
    """
    languages = tuple(batch[0][1].keys())
    operations = batch[0][2][languages[0]].keys()
    firstElement = computeInferenceData_Samples(wordsToOneHots([t[0] for t in batch]).unsqueeze(-1), vocabulary)
    secondElement = formatTargets({lang:[t[1][lang] for t in batch] for lang in languages})
    maxSourceLength = firstElement[2] - 1
    maxCognateLength = {lang: secondElement[lang][2] for lang in MODERN_LANGUAGES}
    lastElement: CachedTargetProbs = {lang: {op:pad2d_sequence([t[2][lang][op] for t in batch], 0)[:maxSourceLength, :maxCognateLength[lang]].squeeze(3) for op in operations} for lang in languages}

    return (firstElement, secondElement, lastElement)


raw_cognates = [{lang: raw_cognates[lang][i] for lang in MODERN_LANGUAGES} for i in range(len(raw_cognates['french']))]
myDataset = MyDataset(raw_samples, raw_cognates, target_probs)
MINI_BATCH_SIZE = 30
dl = DataLoader(dataset = myDataset, batch_size=MINI_BATCH_SIZE, collate_fn=__training__collate_fn, shuffle=True)

randomEditModel.train_models(dl)

#If the code above works, then the ideal code below should works
# dp2 = SequenceWrapper(raw_samples).zip(SequenceWrapper(raw_cognates), SequenceWrapper(target_probs))
# dp2 = get_training_datapipe(dp2, MINI_BATCH_SIZE) #type: ignore
# dl2 = DataLoader2(dp2)