import os


def get_labels(train_file_path: str):
    input_file = open(train_file_path, "r")
    vis = {}
    for line in input_file:
        line = line.strip().split()
        if (len(line) == 0):
            continue
        vis[line[1]] = True

    out_file = open(
        os.path.join(os.path.dirname(train_file_path), "ner_labels.txt"), "w")
    for key, _ in vis.items():
        out_file.write(key + '\n')
