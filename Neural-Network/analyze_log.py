from collections import defaultdict
from pprint import pprint
entries = defaultdict(list)
with open("log.txt", "r") as log_file:
    for i,line in enumerate(log_file.readlines()):
        data = line.split(",")
        params = tuple(data[:-1])
        if i == 0:
            titles = params
            continue
        result = float(data[-1])
        entries[params].append(result)

avged_entries = dict()
for entry,res in entries.items():
    avged_entries[entry] = sum(res) / len(res)

def pretty_print_dict(titles, dictionary):
    print("")
    print(" | ".join(titles) + " |")
    print("-" * (3 * len(titles) + len("".join(titles))))
    for row,avg in dictionary.items():
        for title,entry in zip(titles, row):
            print(entry + " "*(len(title)-len(str(entry))) + " | ",end="")
        print("")

pretty_print_dict(titles, avged_entries)
