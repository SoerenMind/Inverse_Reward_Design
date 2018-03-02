import argparse
import csv
import os
import re
import matplotlib.pyplot as plt

class Experiment(object):
    def __init__(self, params, data):
        self.params = params
        self.data = data

    def __str__(self):
        return 'Experiment: ' + str(self.params)

def maybe_num(x):
    try: return int(x)
    except ValueError:
        try: return float(x)
        except ValueError: return x

def concat(folder, element):
    if folder[-1] == '/':
        return folder + element
    return folder + '/' + element

def get_param_vals(folder_name):
    key_vals = re.finditer(r"([^-]+)=([^-]+)", folder_name)
    result_tuple = tuple(((m.group(1), m.group(2)) for m in key_vals))
    result_dict = { k:maybe_num(v) for k, v in result_tuple }
    return result_tuple, result_dict

def load_experiment(folder):
    with open(concat(folder, 'all choosers-means-.csv'), 'r') as csvfile:
        chooser = csvfile.readline().strip().strip(',')
        reader = csv.DictReader(csvfile)
        first_row = next(reader)
        data = { k:[maybe_num(v)] for k, v in first_row.items() }
        for row in reader:
            for k, v in row.items():
                data[k].append(maybe_num(v))
    return chooser, data

def simplify_keys(experiments):
    keys = list(experiments.keys())
    # A key is a tuple of (k, v) pairs
    first_key = keys[0]

    indices_with_no_variation = []
    indices_with_variation = []
    for index in range(len(first_key)):
        if all(key[index] == first_key[index] for key in keys):
            indices_with_no_variation.append(index)
        else:
            indices_with_variation.append(index)

    def simple_key(key):
        return tuple((key[index] for index in indices_with_variation))

    new_experiments = {simple_key(k):v for k, v in experiments.items()}
    controls = dict([first_key[index] for index in indices_with_no_variation])
    return new_experiments, controls

def load_data(folder):
    folder = concat('data', folder)
    experiments = {}
    for experiment in os.listdir(folder):
        key, params_dict = get_param_vals(experiment)
        chooser, data = load_experiment(concat(folder, experiment))
        if 'choosers' in params_dict:
            assert chooser == params_dict['choosers']
        else:
            key = key + (('choosers', chooser),)
            params_dict['choosers'] = chooser
        experiments[key] = Experiment(params_dict, data)
    experiments, control_var_vals = simplify_keys(experiments)
    changing_vars = [var for var, val in experiments.keys()[0]]
    return experiments, changing_vars, control_var_vals

def graph_all(experiments, all_vars, x_var, dependent_vars, independent_vars,
              controls, folder):
    # TODO(rohinmshah): May want to generalize to two dependent vars
    assert len(dependent_vars) == 1
    y_var = dependent_vars[0]

    control_vars = [var for var, val in controls]
    vars_so_far = [x_var] + dependent_vars + independent_vars + control_vars
    remaining_vars = list(set(all_vars) - set(vars_so_far))
    graphs_data = {}

    for experiment in experiments.values():
        params, data = experiment.params, experiment.data
        if not all(params[k] == v for k, v in controls):
            continue

        key = ','.join(['{0}={1}'.format(k, params[k]) for k in remaining_vars])
        if key not in graphs_data:
            graphs_data[key] = []
        graphs_data[key].append(experiment)

    for key, exps in graphs_data.items():
        graph(exps, x_var, dependent_vars, independent_vars, controls, key, folder)

def graph(experiments, x_var, dependent_vars, independent_vars, controls, other_vals, folder):
    assert len(dependent_vars) == 1
    y_var = dependent_vars[0]

    plt.figure()
    for experiment in experiments:
        params, data = experiment.params, experiment.data
        label = ', '.join([str(params[k]) for k in independent_vars])
        plt.plot(data[x_var], data[y_var], label=label)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    
    subtitle = ','.join(['{0}={1}'.format(k, v) for k, v in controls])
    subtitle = '{0},{1}'.format(subtitle, other_vals).strip(',')
    title = '{0} vs {1} for {2}'.format(y_var, x_var, ', '.join(independent_vars))
    plt.title(title + '\n' + subtitle)
    plt.legend() #loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)

    folder = concat('graph', folder)
    filename = '{0}-with-{1}.png'.format(title, subtitle)
    filename = filename.replace(', ', ',').replace(' ', '-')
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.savefig(concat(folder, filename))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-x', '--x_var', default='iteration')
    parser.add_argument('-d', '--dependent_var', action='append', required=True)
    parser.add_argument('-i', '--independent_var', action='append', required=True)
    parser.add_argument('-c', '--control_var_val', action='append', default=[])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    experiments, all_vars, _ = load_data(args.folder)
    controls = [kv_pair.split('=') for kv_pair in args.control_var_val]
    controls = [(k, maybe_num(v)) for k, v in controls]
    graph_all(experiments, all_vars, args.x_var, args.dependent_var,
              args.independent_var, controls, args.folder)
