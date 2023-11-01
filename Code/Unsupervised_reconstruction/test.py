import time
from time import time
from Tests.createSamples import createSamplesBatch
from data.reconstruction_datasets import samplingDataLoader
from data.getDataset import getCognatesSet
from Source.utils import dl_to_ld

def functionToRun():
    """
    Call here the function
    """
    cognates = dl_to_ld(getCognatesSet())
    samples = createSamplesBatch(len(cognates), 40000)
    dataloader = samplingDataLoader(samples, cognates, (30, 50))
    firstElt = next(iter(dataloader))
    print(len(firstElt[0]))
    print(firstElt[0][0][0].size())
    for elt in dataloader:
        print(elt[1])
    
    
if __name__=="__main__":
    start_time = time()
    functionToRun()
    duration = time() - start_time #in seconds

    print(f'\n\nExecution time : {duration//60} minutes and {duration%60} seconds')