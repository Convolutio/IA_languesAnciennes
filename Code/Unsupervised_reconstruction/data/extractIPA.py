import os
import csv
import json

from typing import Literal, Union

os.chdir(os.path.dirname(__file__))


def printNbIPAFrom_Scrap(filePath: str) -> None:
    """
    Print the number of IPA characters contained in the `scraped_IPA.json` file.

    Args:
        filePath: A string representing the path to the `scraped_IPA.json` file. 
    """
    with open(filePath, 'r', encoding="utf-8") as f:
        IPA: list[dict[str, str]] = json.load(f)
        print(len(IPA))


def extractIPAFrom_Scrap(filePath: str, outputType: Literal['csv', 'set']) -> Union[str, set[str]]:
    """
    Process the IPA data from `scraped_IPA.json` by loading the file, cleaning the data,
    and extracting a set of unique 'symbol' values or writing the data to a CSV file.

    Args:
        filePath: A string representing the path to the `scraped_IPA.json` file.
        outputType: Specifies the output type, either 'csv' for CSV file or 'set' for unique symbols.

    Returns:
        str: A string representing the file path of the created CSV file (default: IPA_characters.csv) if 'csv' is chosen as output type.
        set(str): A set containing unique 'symbol' values if 'set' is chosen as output type.
    """

    if not os.path.exists(filePath):
        raise FileNotFoundError(
            f"The provided path '{filePath}' does not exist.")

    if not filePath.lower().endswith('.json'):
        raise ValueError("The provided file is not a JSON file.")

    if outputType not in ['csv', 'set']:
        raise TypeError("The provided type is not a 'csv' or 'set'.")

    IPA: list[dict[str, str]]
    with open(filePath, 'r', encoding="utf-8") as f:
        IPA = [{k: v.rstrip('\t\n') for k, v in entry.items()}
               for entry in json.load(f)]

    if outputType == 'csv':
        outputPath = 'IPA_characters.csv'

        with open(outputPath, 'w', encoding='UTF32') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=IPA[0].keys())
            writer.writeheader()
            writer.writerows(IPA)

        return outputPath

    if outputType == 'set':
        return {entry['symbol'] for entry in IPA}


def extractIPAFrom_MeloniDb(filePath: str) -> set[str]:
    """
    Extract a set of IPA characters from the Meloni database contained in the `romance-ipa.txt` file.

    Args:
        filePath: A string representing the path to the `romance-ipa.txt` file.

    Returns:
        set(str): A set of IPA characters.
    """
    SEP = '\t'

    with open(filePath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return {char for line in lines[1:] for word in line.split(SEP) for char in word.strip(' \t\n')}


def extractIPAFrom_HeDatabase(pathFolder: str) -> set[str]:
    """
    Extract a set of IPA characters from files ending in `_ipa.txt` in the `recons_data` folder.

    Args:
        filePath: A string representing the path to the `recons_data` folder.

    Returns:
        set(str): A set of IPA characters.
    """
    setIPA = set[str]()

    for filename in os.listdir(pathFolder)+[f'../iteration3_{str(i)}' for i in range(1, 5)]:
        if filename.endswith('_ipa.txt'):
            with open(pathFolder + filename, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    for char in line.strip(' \t\n'):
                        setIPA.add(char)

    return setIPA


def generateVocabularyFile(listIPAChars: list[str], filename: str = 'IPA_characters.txt', SEP: str = ', '):
    """
    Generates a text file named `filename` containing the elements of `listIPAChars`,
    a list of IPA characters, separated by `SEP`.

    Args:
        listIPAChars: A list of IPA characters.
        filename: The name of the output file.
        SEP: The separator between each IPA character written in the file.

    Returns:
        str: A string representing the file path of the created text file (default: 'IPA_characters.txt').
    """
    with open(filename, 'w', encoding='utf-8') as file2write:
        content = ""

        for IPA_char in listIPAChars:
            content += IPA_char + SEP

        file2write.write(content[:-2])


generateVocabularyFile(
    list(extractIPAFrom_HeDatabase('./../recons_data/data')))
