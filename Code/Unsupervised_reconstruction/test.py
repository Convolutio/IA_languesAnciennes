import time
from time import time
from torch import tensor
from Tests.createSamples import createSamplesBatch
from data.reconstruction_datasets import samplingDataLoader
from data.vocab import vocabulary
from data.getDataset import getCognatesSet
from source.utils import dl_to_ld

def functionToRun():
    """
    Call here the function
    """
    cognates = [{lang:tensor(vocabulary(list(d[lang]))) for lang in d} for d in dl_to_ld(getCognatesSet())]
    samples = createSamplesBatch(len(cognates), 40000)
    dataloader = samplingDataLoader(samples, cognates, (30, 100))
    firstElt = next(iter(dataloader))
    print(type(firstElt))
    
    
if __name__=="__main__":
    start_time = time()
    functionToRun()
    duration = time() - start_time #in seconds

    print(f'\n\nExecution time : {duration//60} minutes and {duration%60} seconds')