import csv
import matplotlib.pyplot as plt

masculine_pronouns = ['he', 'him', 'his', 'himself', 'man', 'men', 'male']
feminine_pronouns = ['she', 'her', 'hers', 'herself', 'woman', 'women', 'female']

def count_pronouns(text, pronouns):
    text_split = text.lower().split()
    return sum(text_split.count(pron) for pron in pronouns)

def countAdjetive(csv_name):
    d = {}
    with open(csv_name, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        cols = next(csvreader)
        for row in csvreader:
            prompt = row[0]
            text = row[1]
            prof = row[2]
            adj = row[3]
            if adj not in d:
                d[adj] = {}
            mas_count = count_pronouns(text, masculine_pronouns)
            fem_count = count_pronouns(text, feminine_pronouns)
            if prof not in d[adj]:
                d[adj][prof] = (0,0)
            current_masc_count, current_fem_count = d[adj][prof]
            d[adj][prof] = (current_masc_count + mas_count, current_fem_count + fem_count)
            
    for adj, professions in d.items():
        print(f"'{adj}': {{")
        for prof, (mas_count, fem_count) in professions.items():
            if mas_count == 0 and fem_count == 0:
                ratio = "-"
            elif fem_count == 0:
                ratio = "Inf"
            else:
                ratio = f"{mas_count / fem_count:.3f}"
            
            print(f"    '{prof}': Ratio: {ratio}")
        print("}")       
            
    return d
