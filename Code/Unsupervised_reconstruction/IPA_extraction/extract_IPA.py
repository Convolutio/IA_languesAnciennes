import json
import csv
import os
os.chdir(os.path.dirname(__file__))

#Creating a csv file from the json one
def generateCsvFrom_Scrap():
    with open('scraped_IPA.json', 'r', encoding="utf-8") as file:
        IPA:list[dict[str, str]] = json.load(file)
        for i in range(len(IPA)):
            for key in IPA[i]:
                value = IPA[i][key]
                if value.endswith('\t') or value.endswith('\n'):
                    value = value[:-1]
                IPA[i][key]=value
        with open('IPA_characters.csv','w', encoding='UTF32') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=IPA[0].keys())
            writer.writeheader()
            writer.writerows(IPA)

def extractIPAFrom_Scrap():
    with open('scraped_IPA.json', 'r', encoding="utf-8") as file:
        IPA:list[dict[str, str]] = json.load(file)
        IPA_list = set[str]()
        for i in range(len(IPA)):
            for key in IPA[i]:
                value = IPA[i][key]
                if value.endswith('\t') or value.endswith('\n'):
                    value = value[:-1]
                IPA[i][key]=value
            IPA_list.add(IPA[i]['symbol'])
    return IPA_list

def IPA_charactersNumber():
    with open('scraped_IPA.json', 'r', encoding="utf-8") as file:
        IPA:list[dict[str, str]] = json.load(file)
        print(len(IPA))

def extractIPAFrom_MeloniDatabase():
    IPA_lst = set[str]()
    with open('./../romance-ipa.txt', 'r', encoding='utf-8') as file:
        SEP = '\t'
        lines = file.readlines()
        attrs = lines[0].split(SEP)
        for i in range(1, len(lines)):
            line = lines[i]
            for word in line.split(SEP):
                for char in word:
                    IPA_lst.add(char)
    IPA_lst.remove('\n')
    return IPA_lst

def generateVocabularyFile(IPA_charsList:list[str]):
    with open('IPA_characters.txt', 'w', encoding='utf-8') as file2write:
        content = ""
        for IPA_char in IPA_charsList:
            content += IPA_char + ', '
        file2write.write(content[:-2])

def extractIPAFrom_HeDatabase():
    # extract IPA from _ipa.txt files in the recons_data folder
    IPA_lst = set[str]()
    for filename in os.listdir('./../recons_data/data')+[f'../iteration3_{str(i)}' for i in range(1,5)]:
        if filename.endswith('_ipa.txt'):
            with open('./../recons_data/data/'+filename, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    for char in line:
                        IPA_lst.add(char)
    IPA_lst.remove(' ')
    IPA_lst.remove('\n')
    return IPA_lst

generateVocabularyFile(list(extractIPAFrom_HeDatabase()))