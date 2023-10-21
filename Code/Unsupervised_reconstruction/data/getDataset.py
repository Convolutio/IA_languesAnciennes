from typing import Literal
from models.articleModels import ModernLanguages
from data.vocab import wordsToOneHots
from torch import Tensor

def getCognatesSet() -> dict[ModernLanguages, list[str]]:
    """
    Returns a dictionnary with the raw cognates in padded IntTensors with one-hot indices format (without boundaries).
    """
    cognates: dict[ModernLanguages, list[str]] = {}
    for modernLanguage in ('french', 'spanish', 'portuguese', 'romanian', 'italian'):
        with open(f"./recons_data/data/{modernLanguage.capitalize()}_ipa.txt", "r", encoding="utf-8") as file:
            cognates[modernLanguage] = [line[1:-1] for line in file.readlines()[:-1]]
    return cognates

def getTargetsReconstruction()->list[Tensor]:
    l:list[Tensor] = []
    with open(f"./recons_data/data/latin_ipa.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(len(lines)-1):
            l.append(wordsToOneHots([lines[i][1:-1]]).squeeze(1)) # eliminate the escape and the \n
    return l

def getIteration(i:Literal[1,2,3,4]) -> list[str]:
    """
    Returns the samples from the i-th Bouchard-Côte et al.'s model iteration in a padded IntTensor, with one-hot indexes format (without boundaries).
    """
    iteration:list[str]
    with open(f'./recons_data/iteration3_{str(i)}.txt', 'r', encoding='utf-8') as file:
        iteration = [line[:-1] for line in file.readlines()[:-1]]
    return iteration