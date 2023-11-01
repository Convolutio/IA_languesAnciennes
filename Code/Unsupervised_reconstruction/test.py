import time
from time import time
from Tests.createSamples import createSamplesBatch
from data.reconstruction_datasets import samplingDataLoader
from data.getDataset import getCognatesSet
from Source.utils import dl_to_ld

from torch.utils.data import TensorDataset, DataLoader, ChainDataset

def functionToRun():
    """
    Call here the function
    """
    cognates = dl_to_ld(getCognatesSet())
    samples = createSamplesBatch(len(cognates), 400)
    datasets = [TensorDataset(*samples[30*i:30*(i+1)]) for i in range(len(samples)//30)]
    ds = ChainDataset(datasets)
    dl = DataLoader(ds, batch_size=30, drop_last=False)
    elt = next(iter(dl))
    print(type(elt), len(elt), type(elt[0]), elt[0].size())
    for (i, _) in enumerate(dl):
        pass
    print(i)
    
    # dataloader = samplingDataLoader(list(zip(createSamplesBatch(len(cognates), 400), cognates)))
    # firstElt = next(iter(dataloader))
    # print(len(firstElt[0]))
    # print(firstElt[0][0][0].size())
    
    
if __name__=="__main__":
    start_time = time()
    functionToRun()
    duration = time() - start_time #in seconds

    print(f'\n\nExecution time : {duration//60} minutes and {duration%60} seconds')