from torch import Tensor
import torch.optim as optim
import torch.nn as nn
import torch
from torchtext.vocab import Vocab

import numpy as np

import matplotlib.pyplot as plt

from Types.articleModels import ModernLanguages
from Types.models import InferenceData, TargetInferenceData, SourceInferenceData, PADDING_TOKEN

from Source.editModel import EditModel
from Source.packingEmbedding import PackingEmbedding
from Source.cachedModernData import nextOneHots
from Source.dynamicPrograms import compute_mutation_prob, compute_posteriors
from models import ProbCache

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReconstructionModel(nn.Module):
    """
    Let C be the number of cognates pairs in the unsupervised training dataset.
    Let B be the number of samples given for the inference for each cognates pair.

    # Cognates data caching
    In the model initialisation.

    Input cognates tensor shape: (|y|+2, C, 1)\\
    Input cognates' lengths tensor shape: (C, 1)

    # Inference
    This section is about the following methods:
        - forward dynamic programs

    Input samples tensor shape: (|x|+2, C, B)\\
    Input samples' lengths tensor shape: (C, B)

    # Maximisation
    This section is about the following methods:
        - bacward dynamic programs
        - EditModels training
        - logits and targets rendering (debug)
    
    One sample is selected for each cognate pair so B = 1

    Input samples tensor shape: (|x|+2, C, 1)\\
    Input samples' lengths tensor shape: (C, 1)
    """
    def __init__(self, cognatesSet:dict[ModernLanguages, InferenceData], vocab:Vocab, lstm_input_dim:int, lstm_hidden_dim:int):
        super().__init__()
        self.__languages: tuple[ModernLanguages, ...] = tuple(cognatesSet.keys())
        self.IPA_length = len(vocab)-3
        self.vocab = vocab
        self.shared_embedding_layer = PackingEmbedding(len(vocab), lstm_input_dim, padding_idx=vocab[PADDING_TOKEN], device=device)
        self.__editModels = nn.ModuleDict({
                lang:EditModel(
                    targetInferenceData,
                    lang,
                    self.shared_embedding_layer,
                    vocab,
                    lstm_hidden_dim
                ) for (lang, targetInferenceData) in cognatesSet.items()
            })
        self.cachedBatchSize = len(list(cognatesSet.values())[0][1]) # the size C for the inferences
    
    def getModel(self, language:ModernLanguages)->EditModel:
        return self.__editModels[language] #type: ignore
    
    @property
    def languages(self):
        return self.__languages
    
    def __computeSourceInferenceData(self,samples_:InferenceData) -> SourceInferenceData:
        lengths = samples_[1]+2
        maxLength = samples_[2]+2
        return (self.shared_embedding_layer((
            samples_[0].flatten(start_dim=1), lengths.flatten(), False
        )),
        lengths, maxLength)
    
    #-----------TRAINING----------------
    
    def renderMiniBatchData(self, samples_:InferenceData, prob_targets_cache:dict[ModernLanguages, ProbCache], miniBatchSize:int):
        """
        Returns both the following lists (with the same length and gathered in a tuple):
            * mini batches of data, i.e. a list of tuples with the following elements:
                - source forms' data ready for the running of the embedding conversion. 
                - For each modern language, modern forms' data ready for the running of the inference in the edit model.
                - For each modern language, modern forms encoded in one hot vectors, for the masking.
            * mini batches of target probabilities, i.e. a list of Tensors.
        """
        batchSize = len(samples_[1])
        splits:tuple[list[Tensor], list[Tensor]] = (samples_[0].split(miniBatchSize, 1), samples_[1].split(miniBatchSize))
        samples_miniBatches: list[InferenceData] = [(splits[0][i], splits[1][i], int(torch.max(splits[1][i]).item())) for i in range(len(splits[1]))]
        miniBatchesNumber = len(samples_miniBatches) # = batchSize // MINI_BATCH_SIZE
        
        targets_loads = [torch.empty((0,miniBatchSize, 2*(self.IPA_length+1)), device=device) for _ in range(miniBatchesNumber-1)] # dim = (*, b, 2*(|Σ|+1))
        targets_loads.append(torch.empty((0,batchSize%miniBatchSize, 2*(self.IPA_length+1)), device=device))

        modern_miniBatches: dict[ModernLanguages, list[TargetInferenceData]] = {}
        modernOneHotVectors_miniBatches: dict[ModernLanguages, list[Tensor]] = {}

        for lang in self.__languages:
            model = self.getModel(lang)
            
            modern_splits = (model.cachedTargetInputData[0].split(miniBatchSize, 1), model.cachedTargetInputData[1].split(miniBatchSize))
            modern_miniBatches[lang] = []
            for i in range(len(modern_splits)):
                maxRawLength = int(torch.max(modern_splits[1][i]).item())
                modern_miniBatches[lang].append((
                    modern_splits[0][i][:maxRawLength],
                    modern_splits[1][i],
                    maxRawLength
                ))
            modernOneHotVectors_miniBatches[lang] = [nextOneHots(inpt, self.vocab) for inpt in modern_miniBatches[lang]]
            
            
            for i, (samples_, moderns_, modernOneHots) in enumerate(zip(samples_miniBatches, modern_miniBatches[lang], modernOneHotVectors_miniBatches[lang])):
                maxSourceLength = samples_[2] + 2 # max |x|+2
                maxModernLength = moderns_[2] + 1 # max |y|+2
                thisMiniBatchSize = len(samples_[1]) # = MINI_BATCH_SIZE or (batchSize % MINI_BATCH_SIZE)
                
                mini_targetProbCache = ProbCache(maxSourceLength, maxModernLength, (thisMiniBatchSize, 1))
                mini_targetProbCache.dlt = prob_targets_cache[lang].dlt[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]
                mini_targetProbCache.end = prob_targets_cache[lang].end[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]
                mini_targetProbCache.sub = prob_targets_cache[lang].sub[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]
                mini_targetProbCache.ins = prob_targets_cache[lang].ins[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]



                targets_loads[i] = torch.cat((targets_loads[i], model.get_targets(mini_targetProbCache,
                                                                                  (samples_[1].squeeze(-1)+2, maxSourceLength),
                                                                                  (moderns_[1]+1, maxModernLength),
                                                                                  modernOneHots).flatten(end_dim=1)), dim=0)
        return (
            list(zip(
                samples_miniBatches,
                [dict(zip(modern_miniBatches,t)) for t in zip(*modern_miniBatches.values())],
                [dict(zip(modernOneHotVectors_miniBatches,t)) for t in zip(*modernOneHotVectors_miniBatches.values())]
            )), 
            targets_loads)
    
    def renderLogitsForMiniBatch(self, samples_:InferenceData, modern_miniBatches:dict[ModernLanguages, TargetInferenceData], modernOneHotVectors_miniBatches:dict[ModernLanguages, Tensor]):
        logits_load = torch.empty((0, len(samples_[1]), 2*(self.IPA_length+1)), device=device) # dim = (*, b, 2*(|Σ|+1))
                
        samplesInput:SourceInferenceData = self.__computeSourceInferenceData(samples_)
        
        for lang in self.__languages:
            modern_, modernOneHots = modern_miniBatches[lang], modernOneHotVectors_miniBatches[lang]
            model = self.getModel(lang)
            
            logits_load = torch.cat((logits_load, model.get_logits(samplesInput, modern_, modernOneHots).flatten(end_dim=1)), dim=0)
        return logits_load
    
    def train_models(self, samples_:InferenceData, prob_targets_cache:dict[ModernLanguages, ProbCache], miniBatchSize = 30, epochs=5, learning_rate=0.01):
        
        ## Mini-batching
        numberOfMiniBatches = len(samples_[1]) // miniBatchSize  +  int(len(samples_[1]) % miniBatchSize != 0)
        (miniBatches_data, targets_loads) = self.renderMiniBatchData(samples_, prob_targets_cache, miniBatchSize)
        
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #TODO: experimenting smoothing values for this loss function.
        loss_function = nn.KLDivLoss(reduction="batchmean", log_target=True) # neutral if log(target) = log(logit) = 0
        
        trainingStats = {'average':np.zeros(epochs), 'std':np.zeros(epochs)}

        self.train()

        for epochNumber in range(epochs):
            print(f"Epoch {epochNumber+1}\n" + '-'*60)

            losses = torch.zeros(numberOfMiniBatches)

            for i, (miniBatch_data, targets_load) in enumerate(zip(miniBatches_data, targets_loads)):
                optimizer.zero_grad()
                
                logits_load = self.renderLogitsForMiniBatch(*miniBatch_data)
                
                loss = loss_function(logits_load, targets_load)
                loss.backward()
                optimizer.step()

                losses[i] = loss
                print(f'loss: {loss.item():>7f} [minibatch {i+1}/{numberOfMiniBatches}]'+' '*10, end='\r')
            
            mean_loss, std_loss = losses.mean().item(), losses.std().item()
            trainingStats['average'][epochNumber] = mean_loss
            trainingStats['std'][epochNumber] = std_loss
            print(f'Average loss: {mean_loss:>7f} ; Standard deviation: {std_loss:>7f}' + ' '*10+'\n'+'-'*60+'\n')

        fig, ax = plt.subplots()
        ax.errorbar(np.arange(1,6), trainingStats['average'], trainingStats['std'],
                    marker='o', markerfacecolor='tab:red', markeredgecolor="tab:red", ecolor='red', color='tab:orange')
        ax.set_title("Evolution of the computed loss according to the training epoch.")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Loss value (KLDivLoss)")
        print(fig)






    #------------INFERENCE------------------
    
    def update_modernForm_context(self):
        self.eval()
        with torch.no_grad():
            for language in self.languages:
                model = self.getModel(language)
                model.update_cachedTargetContext()
    
    def forward_dynProg(self, sources_:InferenceData) -> dict[ModernLanguages, Tensor]:
        """
        Returns (p(y_l)|x) for all the batch
        """
        self.eval()
        with torch.no_grad():
            MINI_BATCH_SIZE = 2
            sources:list[SourceInferenceData] = [self.__computeSourceInferenceData((*s, sources_[2])) for s in zip(sources_[0].split(MINI_BATCH_SIZE, -1), sources_[1].split(MINI_BATCH_SIZE, -1))]

            probs:dict[ModernLanguages, Tensor] = {}
            for language in self.languages:
                model = self.getModel(language)
                mutation_prob:Tensor = compute_mutation_prob(model, sources, model.cachedTargetDataForDynProg) #type: ignore
                probs[language] = mutation_prob
            
            return probs
    
    def backward_dynProg(self, sources_: InferenceData) -> dict[ModernLanguages, ProbCache]:
        self.eval()
        assert(sources_[0].size()[1:]==(self.cachedBatchSize,1)), "The expected shape is (C,1), with C the number of cognates pairs passed in the initialisation of the model."
        with torch.no_grad():
            sources:SourceInferenceData = self.__computeSourceInferenceData(sources_)

            cache:dict[ModernLanguages, ProbCache] = {}
            for language in self.languages:
                model = self.getModel(language)
                cache[language] = compute_posteriors(model, [sources], model.cachedTargetDataForDynProg)
                
            return cache