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
    """Converts string x to an int if possible, otherwise a float if possible,
    otherwise returns it unchanged."""
    try: return int(x)
    except ValueError:
        try: return float(x)
        except ValueError: return x

def concat(folder, element):
    """folder and element are strings"""
    if folder[-1] == '/':
        return folder + element
    return folder + '/' + element

def get_param_vals(folder_name):
    """Gets the parameter values of the experiment from its folder name.

    folder_name is a string such as "2018-03-02 13:17:51.-beta=0.1-dim=10-dist_scale=0.5-gamma=1.0-mdp=gridworld-num_exp=100-num_iter=20-num_iters_optim=10-num_q_max=500-num_states=100-num_traject=1-qsize=10-seed=1-size_proxy=100-size_true=10000-value_iters=25"
    Returns two things:
    - A tuple of tuples of strings, of the form ((key, value), ...)
    - A dictionary mapping strings to strings or numbers, of the form
      {key : value, ...}
    """
    key_vals = re.finditer(r"([^-]+)=([^-]+)", folder_name)
    result_tuple = tuple(((m.group(1), m.group(2)) for m in key_vals))
    result_dict = { k:maybe_num(v) for k, v in result_tuple }
    return result_tuple, result_dict

def load_experiment(folder):
    """Loads the data from <folder>/all choosers-means-.csv.

    Returns two things:
    - chooser: A string, which chooser was used for this experiment
    - data: Dictionary mapping keys (such as test_entropy) to lists of numbers.
    """
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
    """Identifies experiment parameters that are constant across the dataset and
    removes them from the keys, leaving shorter, simpler keys.

    experiments: Dictionary from keys of the form ((var, val), ...) to
        Experiment objects
    Returns two things:
      - new_experiments: Same type as experiments, but with smaller keys
      - controls: Dictionary of the form {var : val} containing the parameters and
          their values that did not change over the experiments.
    """
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
    """Loads all experiment data from data/<folder>.

    Returns three things:
    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects
    - changing_vars: List of strings, the variables that have more than one
          distinct value across Experiments
    - control_var_vals: Dictionary of the form {var : val} containing the
          parameters and their values that did not change over the experiments.
    """
    folder = concat('data', folder)
    experiments = {}
    for experiment in os.listdir(folder):
        if not experiment.startswith('2018'):
            continue
        key, params_dict = get_param_vals(experiment)
        chooser, data = load_experiment(concat(folder, experiment))
        if 'choosers' in params_dict:
            assert chooser == params_dict['choosers']
        else:
            key = key + (('choosers', chooser),)
            params_dict['choosers'] = chooser
        experiments[key] = Experiment(params_dict, data)
    experiments, control_var_vals = simplify_keys(experiments)
    experiments = fix_special_cases(experiments)
    changing_vars = [var for var, val in experiments.keys()[0]]
    return experiments, changing_vars, control_var_vals

def graph_all(experiments, all_vars, x_var, dependent_vars, independent_vars,
              controls, folder):
    """Graphs data and saves them.

    Each graph generated plots the dependent_vars against x_var for all
    valuations of independent_vars, with the control variables set to the values
    specified in controls. For every valuation of variables not in x_var,
    dependent_vars, independent_vars, or controls, a separate graph is
    generated and saved in folder.

    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects
    - all_vars: List of strings, all the variables that have some variation
    - x_var: Variable that provides the data for the x-axis
    - dependent_vars: List of strings, variables to plot on the y-axis
    - independent_vars: List of strings, experimental conditions to plot on the
          same graph
    - controls: Tuple of the form ((var, val), ...) where var is a string and
          val is a string or number. The values of control variables.
    - folder: Graphs are saved to graph/<folder>/
    """
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

def graph(experiments, x_var, dependent_vars, independent_vars, controls,
          other_vals, folder):
    """Creates and saves a single graph.

    Arguments are almost the same as for graph_all.
    - other_vals: String of the form "{var}={val},..." specifying values of
          variables not in x_var, dependent_vars, independent_vars, or
          controls.
    """
    assert len(dependent_vars) in [1, 2]

    def make_graph(ax, y_var):
        for experiment in experiments:
            params, data = experiment.params, experiment.data
            label = ', '.join([str(params[k]) for k in independent_vars])
            ax.plot(data[x_var], data[y_var], label=label)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)

    fig, ax1 = plt.subplots()
    make_graph(ax1, dependent_vars[0])
    if len(dependent_vars) == 2:
        ax2 = ax1.twinx()
        make_graph(ax2, dependent_vars[1])

    subtitle = ','.join(['{0}={1}'.format(k, v) for k, v in controls])
    subtitle = '{0},{1}'.format(subtitle, other_vals).strip(',')
    title = 'Data for {0}'.format(', '.join(independent_vars))
    plt.title(title + '\n' + subtitle)
    plt.legend() #loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    fig.tight_layout()

    folder = concat('graph', folder)
    filename = '{0}-vs-{1}-for-{2}-with-{3}.png'.format(
        ','.join(dependent_vars), x_var, ','.join(independent_vars), subtitle)
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
