from json import load
import csv
from os import chdir, path
chdir(path.dirname(__file__))

#Creating a csv file from the json one
def generateCsv_toScrap():
    with open('scraped_IPA.json', 'r', encoding="utf-8") as file:
        IPA:list[dict[str, str]] = load(file)
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

def IPA_charactersNumber():
    with open('scraped_IPA.json', 'r', encoding="utf-8") as file:
        IPA:list[dict[str, str]] = load(file)
        print(len(IPA))

def extractIPAFromDatabase():
    IPA_lst = set()
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
    with open('IPA_characters.txt', 'w', encoding='utf-8') as file2write:
        content = ""
        for elt in IPA_lst:
            content += elt + ', '
        file2write.write(content[:-2])
    return IPA_lst

print(extractIPAFromDatabase())

#generateCsv_toScrap()