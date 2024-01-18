import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from torch.types import Device
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab

from models.types import (ModernLanguages, Operations, InferenceData_Samples,
                          InferenceData_Cognates, InferenceData_SamplesEmbeddings,
                          PADDING_TOKEN)


from Source.editModel import EditModel
from Source.packingEmbedding import PackingEmbedding
from Source.dynamicPrograms import compute_mutation_prob, compute_posteriors
from models.probcache import ProbCache

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ReconstructionModel(nn.Module):
    """
    #TODO: update the documentation (for now, don't read it).

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
        - backward dynamic programs
        - EditModels training
        - logits and targets rendering (debug)

    One sample is selected for each cognate pair so B = 1

    Input samples tensor shape: (|x|+2, C, 1)\\
    Input samples' lengths tensor shape: (C, 1)
    """

    def __init__(self, languages: tuple[ModernLanguages, ...], vocab: Vocab, lstm_input_dim: int, lstm_hidden_dim: int, device: Device = device):
        super().__init__()
        self.__languages = languages
        self.IPA_length = len(vocab)-3
        self.vocab = vocab
        self.shared_embedding_layer = PackingEmbedding(
            len(vocab), lstm_input_dim, padding_idx=vocab[PADDING_TOKEN], device=device)
        self.__editModels = nn.ModuleDict({
            lang: EditModel(
                lang,
                self.shared_embedding_layer,
                vocab,
                device,
                lstm_hidden_dim
            ) for lang in languages
        })

    def getModel(self, language: ModernLanguages) -> EditModel:
        return self.__editModels[language]  # type: ignore

    @property
    def languages(self):
        return self.__languages

    def __computeSourceInferenceData(self, samples_: InferenceData_Samples) -> InferenceData_SamplesEmbeddings:
        lengths = samples_[1]
        maxLength = samples_[2]
        return (self.shared_embedding_layer(
            (samples_[0].flatten(start_dim=1), lengths.flatten(), False)),
            lengths, maxLength)

    # -----------TRAINING----------------
    def train_models(self, training_data_loader: DataLoader[tuple[
        InferenceData_Samples,
        dict[ModernLanguages, InferenceData_Cognates],
        dict[ModernLanguages, dict[Operations, Tensor]]
    ]], epochs: int = 5, learning_rate: float = 0.01):
        """
        Arguments:
            training_set: a dataload of tuples containing (
                the sampled reconstructions,
                their cognates,
                its targets edit probabilities (shape = (|x|+1, |y|+1, c, 1))
                )
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # TODO: experimenting smoothing values for this loss function.
        # neutral if log(target) = log(logit) = 0
        loss_function = nn.KLDivLoss(reduction="batchmean", log_target=True)

        trainingStats = {'average': np.zeros(epochs), 'std': np.zeros(epochs)}

        self.train()

        for epochNumber in range(epochs):
            print(f"Epoch {epochNumber+1}\n" + '-'*60)

            losses = []
            for (i, (sources, cognates, targets_load)) in enumerate(training_data_loader):
                optimizer.zero_grad()
                # shape = ((|x|+1)*(|y|+1), c, 4)
                targets = torch.cat([torch.cat((
                    targets_load[lang]['sub'],
                    targets_load[lang]['dlt'],
                    targets_load[lang]['ins'],
                    targets_load[lang]['end'],
                ), dim=-1).flatten(end_dim=1) for lang in self.languages], dim=0)

                logits_load = []
                for lang in self.languages:
                    logits = self.getModel(lang).forward_and_select(
                        self.__computeSourceInferenceData(sources),
                        cognates[lang]
                    )
                    logits = torch.cat((
                        logits['sub'],
                        logits['dlt'],
                        logits['ins'],
                        logits['end']
                    ), dim=-1).flatten(end_dim=1)
                    logits_load.append(logits)

                # shape = ((|x|+1)*(|y|+1), c, 4)
                logits = torch.cat(logits_load, dim=0)
                logits_load = []

                loss = loss_function(logits, targets)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                print(f'loss: {loss.item():>7f} [minibatch {i+1}]'+' '*10, 
                      end='\r')

            losses = torch.tensor(losses)
            mean_loss, std_loss = losses.mean().item(), losses.std().item()
            trainingStats['average'][epochNumber] = mean_loss
            trainingStats['std'][epochNumber] = std_loss
            print(f'Average loss: {mean_loss:>7f} ; Standard deviation: {std_loss:>7f}' + ' '*10+'\n'+'-'*60+'\n')

        fig, ax = plt.subplots()
        ax.errorbar(np.arange(1, 6), trainingStats['average'], trainingStats['std'],
                    marker='o', markerfacecolor='tab:red', markeredgecolor="tab:red", 
                    ecolor='red', color='tab:orange')
        ax.set_title("Evolution of the computed loss according to the training epoch.")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Loss value (KLDivLoss)")
        print(fig)

    # ------------INFERENCE------------------
    def forward_dynProg(self, samples: InferenceData_Samples, cognates: dict[ModernLanguages, InferenceData_Cognates]):
        """
        Computes (p(y_l)|x) for all the batch (save the results in file)
        """
        self.eval()

        probs: dict[ModernLanguages, Tensor] = {}
        with torch.no_grad():
            sources: InferenceData_SamplesEmbeddings = self.__computeSourceInferenceData(
                samples)

            for language in self.languages:
                model: EditModel = self.__editModels[language]  # type: ignore
                mutation_prob = compute_mutation_prob(
                    model, sources, cognates[language])
                probs[language] = mutation_prob  # type: ignore

        return probs

    def backward_dynProg(self, sources_: InferenceData_Samples, targets_: dict[ModernLanguages, InferenceData_Cognates]) -> dict[ModernLanguages, ProbCache]:
        self.eval()

        with torch.no_grad():
            sources: InferenceData_SamplesEmbeddings = self.__computeSourceInferenceData(
                sources_)

            cache: dict[ModernLanguages, ProbCache] = {}
            for language in self.languages:
                model: EditModel = self.__editModels[language]  # type: ignore
                cache[language] = compute_posteriors(
                    model, sources, targets_[language])

            return cache

    # TODO: enlever
    def infer(self, sources_: InferenceData_Samples, targets_: dict[ModernLanguages, InferenceData_Cognates]):
        model: EditModel = self.getModel("french")  # type:ignore
        sub_results, ins_results = model.forward_and_select(
            self.__computeSourceInferenceData(sources_), targets_['french'])
        return sub_results, ins_results
