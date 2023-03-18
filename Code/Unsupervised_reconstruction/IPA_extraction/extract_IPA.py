from json import load
import csv

#Creating a csv file from the json one
def generateCsv_toScrap():
    with open('IPA_extraction\\scraped_IPA.json', 'r', encoding="utf-8") as file:
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
    with open('IPA_extraction\\scraped_IPA.json', 'r', encoding="utf-8") as file:
        IPA:list[dict[str, str]] = load(file)
        print(len(IPA))

generateCsv_toScrap()