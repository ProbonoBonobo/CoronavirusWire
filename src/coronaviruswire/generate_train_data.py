"""
Partition a local CSV file (downloaded from the Google Sheets url) into
training/validation sets and convert headers into the format required
by our BERT multilabel classification model.
"""
import numpy as np
import csv
from src.coronaviruswire.utils import load_csv
from src.coronaviruswire.common import db

#  use a 80%/20% partition of training/validation examples
validation_pct = 0.2
train_pct = 1 - validation_pct
label_dimensions = ("audience_tag", "is_relevant")
ignored_dimensions = ("mentions_covid19?",)
crawldb = db["crawldb"]


def load_samples(
    fp="/home/kz/projects/coronaviruswire/lib/annotated/multilabel_training_set_v5.csv"
):
    data = load_csv(fp, delimiter=",")
    samples = [
        row
        for row in data
        if all(row[dim] not in (None, "") for dim in label_dimensions)
    ]
    print(f"Loaded {len(samples)} samples.")
    return samples


def extract_labels(data):
    labels = {dim: set() for dim in label_dimensions}
    for sample in data:
        for dim in label_dimensions:
            labels[dim].add(sample[dim])
    print(labels)
    return labels


def transform_columns(data, labels):
    label_names = []
    for dim_name, values in labels.items():
        if all(v in "01" for v in values):
            label_names.append(dim_name)
        else:
            label_names.extend(list(values))
    print(label_names)
    training_set = []
    for i, row in enumerate(data):
        text = []
        _tx = {}
        for col, value in row.items():
            if not col or col in ignored_dimensions:
                continue
            elif col and col not in label_dimensions:
                text.append(f"{col}: {value}")
            elif col in label_names:
                _tx[col] = int(value)
            else:
                for option in labels[col]:
                    _tx[option] = int(option == value)
        _tx["id"] = i
        _tx["text"] = "\n".join(text)
        tx = (_tx["id"], _tx["text"], *[_tx[col] for col in label_names])
        training_set.append(tx)
    col_names = ["id", "text", *label_names]
    return col_names, training_set


def merge_input_columns(data):
    for row in data:
        pass


def write_labels_csv(labels, fp="../../lib/fast-bert/labels/labels.csv"):
    print(f"Labels: {labels}")
    with open(fp, "w") as f:
        for label in labels:
            f.write(label)
            f.write("\n")


def write_samples(
    cols,
    rows,
    train_fp="../../lib/fast-bert/data/train.csv",
    valid_fp="../../lib/fast-bert/val.csv",
):
    k = len(rows)

    _train = open(train_fp, "w")
    _valid = open(valid_fp, "w")

    train = csv.writer(_train)
    valid = csv.writer(_valid)

    train.writerow(cols)
    valid.writerow(cols)

    for row, destination in zip(
        rows, np.random.choice([train, valid], size=k, p=[train_pct, validation_pct])
    ):

        destination.writerow(row)
    _train.close()
    _valid.close()


if __name__ == "__main__":
    data = load_samples()
    labels = extract_labels(data)
    cols, rows = transform_columns(data, labels)
    write_labels_csv(list(labels.keys()))
    write_samples(cols, rows)
