from torch import Tensor
import torch.optim as optim
import torch.nn as nn
import torch

import numpy as np

import matplotlib.pyplot as plt

from Types.articleModels import ModernLanguages
from Types.models import InferenceData, TargetInferenceData, SourceInferenceData


from Source.editModel import EditModel
from Source.packingEmbedding import PackingEmbedding
from Source.cachedModernData import nextOneHots
from Source.dynamicPrograms import compute_mutation_prob, compute_posteriors
from models import ProbCache


class ReconstructionModel(nn.Module):
    def __init__(self, cognatesSet:dict[ModernLanguages, InferenceData], voc_size:int, lstm_input_dim:int, lstm_hidden_dim:int):
        super().__init__()
        self.__languages: tuple[ModernLanguages, ...] = tuple(cognatesSet.keys())
        self.__voc_size = voc_size
        self.shared_embedding_layer = PackingEmbedding(voc_size+3, lstm_input_dim, padding_idx=0)
        self.__editModels = nn.ModuleDict({
                lang:EditModel(
                    targetInferenceData,
                    lang,
                    self.shared_embedding_layer,
                    voc_size,
                    lstm_hidden_dim
                ) for (lang, targetInferenceData) in cognatesSet.items()
            })
    
    def getModel(self, language:ModernLanguages):
        return self.__editModels[language]
    
    @property
    def languages(self):
        return self.__languages
    
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
        
        targets_loads = [torch.empty((0,miniBatchSize, 2*(self.__voc_size+1))) for _ in range(miniBatchesNumber-1)] # dim = (*, b, 2*(|Σ|+1))
        targets_loads.append(torch.empty((0,batchSize%miniBatchSize, 2*(self.__voc_size+1))))

        modern_miniBatches: dict[ModernLanguages, list[TargetInferenceData]] = {}
        modernOneHotVectors_miniBatches: dict[ModernLanguages, list[Tensor]] = {}

        for lang in self.__languages:
            model:EditModel = self.getModel(lang) #type: ignore
            
            modern_splits = (model.cachedTargetInputData[0].split(miniBatchSize, 1), model.cachedTargetInputData[1].split(miniBatchSize))
            modern_miniBatches[lang] = [(modern_splits[0][i], modern_splits[1][i], int(torch.max(modern_splits[1][i]).item())) for i in range(len(modern_splits[1]))]
            modernOneHotVectors_miniBatches[lang] = [nextOneHots(inpt, self.__voc_size) for inpt in modern_miniBatches[lang]]
            
            
            for i, (samples_, moderns_, modernOneHots) in enumerate(zip(samples_miniBatches, modern_miniBatches[lang], modernOneHotVectors_miniBatches[lang])):
                maxSourceLength = samples_[2] + 2 # max |x|+2
                maxModernLength = moderns_[2] + 1 # max |y|+2
                thisMiniBatchSize = len(samples_[1]) # = MINI_BATCH_SIZE or (batchSize % MINI_BATCH_SIZE)
                
                mini_targetProbCache = ProbCache(maxSourceLength, maxModernLength, thisMiniBatchSize)
                mini_targetProbCache.dlt = prob_targets_cache[lang].dlt[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]
                mini_targetProbCache.end = prob_targets_cache[lang].end[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]
                mini_targetProbCache.sub = prob_targets_cache[lang].sub[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]
                mini_targetProbCache.ins = prob_targets_cache[lang].ins[:maxSourceLength, :maxModernLength, miniBatchSize*i : miniBatchSize*i + thisMiniBatchSize]



                targets_loads[i] = torch.cat((targets_loads[i], model.get_targets(mini_targetProbCache,
                                                                                  (samples_[1]+2, maxSourceLength),
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
        logits_load = torch.empty((0, len(samples_[1]), 2*(self.__voc_size+1))) # dim = (*, b, 2*(|Σ|+1))
                
        samplesInput:SourceInferenceData = (self.shared_embedding_layer((samples_[0], samples_[1]+2, False)), samples_[1]+2, samples_[2]+2)
        
        for lang in self.__languages:
            modern_, modernOneHots = modern_miniBatches[lang], modernOneHotVectors_miniBatches[lang]
            model:EditModel = self.getModel(lang) #type: ignore
            
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
            print(f'Average loss: {mean_loss:>7f} ; Standard deviation: {std_loss:>7f}' + ' '*10 + '\n')

        fig, ax = plt.subplots()
        ax.errorbar(np.arange(1,6), trainingStats['average'], trainingStats['std'],
                    marker='o', markerfacecolor='tab:red', markeredgecolor="tab:red", ecolor='red', color='tab:orange')
        ax.set_title("Evolution of the computed loss according to the training epoch.")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Loss value (KLDivLoss)")
        print(fig)






    #------------INFERENCE------------------
    
    def update_modernForm_context(self):
        for language in self.languages:
            model:EditModel = self.__editModels[language] #type: ignore
            model.update_cachedTargetContext()
    
    def forward_dynProg(self, sources_:InferenceData) -> dict[ModernLanguages, np.ndarray]:
        """
        Returns (p(y_l)|x) for all the batch
        """
        with torch.no_grad():
            lengths = sources_[1]+2
            maxLength = sources_[2]+2
            sources:SourceInferenceData = (self.shared_embedding_layer((sources_[0], lengths, False)), lengths, maxLength)

            probs:dict[ModernLanguages, np.ndarray] = {}
            for language in self.languages:
                model:EditModel = self.__editModels[language] #type: ignore
                mutation_prob = compute_mutation_prob(model, sources, model.cachedTargetDataForDynProg)
                probs[language] = mutation_prob #type: ignore
            
            return probs
    
    def bacward_dynProg(self, sources_: InferenceData) -> dict[ModernLanguages, ProbCache]:
        with torch.no_grad():
            lengths = sources_[1]+2
            maxLength = sources_[2]+2
            sources:SourceInferenceData = (self.shared_embedding_layer((sources_[0], lengths, False)), lengths, maxLength)

            cache:dict[ModernLanguages, ProbCache] = {}
            for language in self.languages:
                model:EditModel = self.__editModels[language] #type: ignore
                cache[language] = compute_posteriors(model, sources, model.cachedTargetDataForDynProg)
                
            return cache