from collections import defaultdict
import sys

def print_table_from_file(filename, sortby = None):
    titles, avged_entries = setup(filename)
    build_table(titles, avged_entries, sortby=sortby)


def setup(filename):
    entries = defaultdict(list)
    with open("log.txt", "r") as log_file:
        for i,line in enumerate(log_file.readlines()):
            data = line.split(",")
            params = data[:-1]
            if i == 0:
                params.insert(0,"count")
                titles = params
                continue
            params = tuple(params)
            result = float(data[-1])
            entries[params].append(result)

    avged_entries = dict()
    for entry,res in entries.items():
        count = len(res)
        new_entry = list(entry)
        new_entry.insert(0, "({0})".format(str(len(res))))
        new_entry = tuple(new_entry)
        avged_entries[new_entry] = sum(res) / len(res)
    return titles, avged_entries

def build_table(titles, dictionary, sortby=None):
    ''' 
    Print dictionary in human-readable table format
    titles - a list of column headers for each key
    dictionary - maps a string of column names (comma-separated) mapping to a value
    sortby - If None, it sorts by the dictionary value, elsewise sorts by the 
    column header specified
    '''
    print("")
    print(" | ".join(titles) + " | accuracy")
    print("-" * (3 * len(titles) + len("".join(titles)) + 8))
    if sortby == None:
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
    else:
        key_index = titles.index(sortby)
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[0][key_index])
    for row,avg in sorted_dict:
        for title,entry in zip(titles, row):
            print(entry + " "*(len(title)-len(str(entry))) + " | ",end="")
        print(str(avg))
        #print("")
    print("")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sort = sys.argv[1]
    else: sort = None
    print_table_from_file("log.txt", sortby=sort)
