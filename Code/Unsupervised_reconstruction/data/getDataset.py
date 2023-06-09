from typing import Literal
from Types.articleModels import CognatesSet

def getCognatesSet() -> CognatesSet:
    cognates:CognatesSet = {"french":[], "spanish":[], "portuguese":[], "italian":[], "romanian":[]}
    for modernLanguage in cognates:
        with open(f"./recons_data/data/{modernLanguage.capitalize()}_ipa.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
            for i in range(len(lines)-1):
                cognates[modernLanguage].append(lines[i][1:-1]) # eliminate the escape and the \n
    return cognates

def getTargetsReconstruction()->list[str]:
    l:list[str] = []
    with open(f"./recons_data/data/latin_ipa.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(len(lines)-1):
            l.append(lines[i][1:-1]) # eliminate the escape and the \n
    return l

def getIteration(i:Literal[1,2,3,4]) -> list[str]:
    """
    Returns the i-th Bouchard's iteration.
    """
    iteration = []
    with open(f'./recons_data/iteration3_{str(i)}.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for j in range(len(lines)-1):
            iteration.append(lines[j][:-1])
    return iteration