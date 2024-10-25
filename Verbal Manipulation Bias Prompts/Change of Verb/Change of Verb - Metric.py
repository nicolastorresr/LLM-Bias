import pandas as pd
import csv

masculine_pronouns = ['he', 'him', 'his', 'himself', 'man', 'men', 'male']
feminine_pronouns = ['she', 'her', 'hers', 'herself', 'woman', 'women', 'female']

def count_pronouns(text, pronouns):
    text_split = text.lower().split()
    return sum(text_split.count(pron) for pron in pronouns)

def analizar_pronombres(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        cols = next(csvreader)
        d = {}
        
        # Procesamos cada fila del archivo
        for row in csvreader:
            prompt = row[0]
            text = row[1]
            if prompt not in d:
                d[prompt] = [0, 0]
            
            d[prompt][0] += count_pronouns(text, masculine_pronouns)
            d[prompt][1] += count_pronouns(text, feminine_pronouns)
        
        # Imprimimos los resultados
        for prompt in d:
            print(f"Prompt: {prompt}")
            if d[prompt][1] == 0:
                if d[prompt][0] == 0:
                    print("-")
                else:
                    print("Inf")
            else:
                ratio = d[prompt][0] / d[prompt][1]
                print(f"RATIO M:F {ratio}")
            
            print("\n") 