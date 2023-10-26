from torch import load as tload
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2 import MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper
from data.datapipes import samplingDataPipe

languages = ['french', 'spanish', 'italian', 'portuguese', 'romanian']
cognates = IterableWrapper([{language:cognate for language in languages} for cognate in ["bizɑ̃tˈɛ̃",
 "bizˈɔ̃",
 "blasfˈɛm",
 "bˈœf",
 "boʁˈaks",
 "boʁeˈal",
 "batwajˈe",
 "bovˈɛ̃",
 "bʁaʃjˈal",
 "bʁˈa",
]])
# list of C samples tensors of shape (L~, B)
samples = IterableWrapper(tload('./tests/samples.pt'))

dp = samplingDataPipe(samples, cognates, (10, 100)) #type: ignore
mp_serv = MultiProcessingReadingService(num_workers=4)
dl = DataLoader2(datapipe=dp, reading_service=mp_serv) #type: ignore
iterator = iter(dp)
for inpt in iterator:
    print(type(inpt[0]))
    print(type(inpt[1]["french"]))
    print(inpt[2])
    print()