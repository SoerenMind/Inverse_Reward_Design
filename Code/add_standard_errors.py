import csv
import numpy as np
import os
import sys

from scipy.stats import sem

def maybe_num(x):
    """Converts string x to an int if possible, otherwise a float if possible,
    otherwise returns it unchanged."""
    x = x.strip('[').strip(']')
    try: return int(x)
    except ValueError:
        try: return float(x)
        except ValueError: return x

def concat(folder, element):
    """folder and element are strings"""
    if folder[-1] == '/':
        return folder + element
    return folder + '/' + element

def load_one(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        first_row = next(reader)
        data = { k:[maybe_num(v)] for k, v in first_row.items() }
        for row in reader:
            for k, v in row.items():
                data[k].append(maybe_num(v))
    return reader.fieldnames, data

def load_experiment(folder):
    with open(concat(folder, 'all choosers-means-.csv'), 'r') as csvfile:
        chooser = csvfile.readline().strip().strip(',')

    experiment_data = []
    for experiment in os.listdir(folder):
        if experiment.endswith('-.csv'):
            continue
        keys, data = load_one(concat(folder, experiment))
        experiment_data.append(data)

    def get_data(k):
        return np.array([data[k] for data in experiment_data])

    all_data = { k : get_data(k) for k in experiment_data[0].keys() }
    return chooser, keys, all_data

def compute_standard_errors(data):
    iteration_data = data['iteration'][0]
    result = { k:sem(v, axis=0) for k, v in data.items() }
    result['iteration'] = iteration_data
    return result

def write_standard_errors(folder, chooser, keys, standard_errors):
    with open(concat(folder, 'all choosers-sterr-.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writerow({'iteration': chooser})
        writer.writeheader()
        for i in range(len(standard_errors['iteration'])):
            writer.writerow({k:v[i] for k, v in standard_errors.items()})

def handle_experiment(folder):
    chooser, keys, all_data = load_experiment(folder)
    standard_errors = compute_standard_errors(all_data)
    write_standard_errors(folder, chooser, keys, standard_errors)

def fix_all(folder):
    for subfolder in os.listdir(folder):
        if not subfolder.startswith('2018'):
            continue
        handle_experiment(concat(folder, subfolder))

if __name__ == '__main__':
    fix_all(sys.argv[1])
