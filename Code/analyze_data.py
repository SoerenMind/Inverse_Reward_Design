
print('importing')
import argparse
import csv
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from seaborn import set, set_style
import numpy as np

print('importing done')

class Experiment(object):
    def __init__(self, params, means_data, sterrs_data):
        self.params = params
        self.means_data = means_data
        self.sterrs_data = sterrs_data

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
    - A tuple of tuples of strings/numbers, of the form ((key, value), ...)
    - A dictionary mapping strings to strings or numbers, of the form
      {key : value, ...}
    """
    key_vals = re.finditer(r"([^-]+)=([^-]+)", folder_name)
    result_tuple = tuple(((m.group(1), maybe_num(m.group(2))) for m in key_vals))
    result_dict = { k:v for k, v in result_tuple }
    return result_tuple, result_dict

def load_experiment_file(filename):
    """Loads the data from <filename>, which should be an all choosers CSV.

    Returns two things:
    - chooser: A string, which chooser was used for this experiment
    - data: Dictionary mapping keys (such as test_entropy) to lists of numbers.
    """
    with open(filename, 'r') as csvfile:
        chooser = csvfile.readline().strip().strip(',')
        reader = csv.DictReader(csvfile)
        first_row = next(reader)
        data = { k:[maybe_num(v)] for k, v in first_row.items() }
        for row in reader:
            for k, v in row.items():
                data[k].append(maybe_num(v))
    return chooser, data

def load_experiment(folder):
    """Loads the data from <folder>, specifically from all choosers-means-.csv
    and all choosers-sterr-.csv.

    Returns three things:
    - chooser: A string, which chooser was used for this experiment
    - means_data: Dictionary mapping keys to lists of numbers (means).
    - sterr_data: Dictionary mapping keys to lists of numbers (std errors).
    """
    means_chooser, means_data = load_experiment_file(
        concat(folder, 'all choosers-means-.csv'))
    sterrs_chooser, sterrs_data = load_experiment_file(
        concat(folder, 'all choosers-sterr-.csv'))
    assert means_chooser == sterrs_chooser
    return means_chooser, means_data, sterrs_data

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

def fix_special_cases(experiments):
    """Does postprocessing to handle any special cases.

    For example, the full chooser is only run with query size 2 since it doesn't
    depend on query size, but we want it to be plotted with all the other
    queries as well, so we duplicate the experiment with other query sizes.

    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects
    """
    query_sizes = set([exp.params['qsize'] for exp in experiments.values()])
    full_exps = [(key, exp) for key, exp in experiments.items() if exp.params['choosers'] == 'full']
    def replace(key_tuple, var, val):
        return tuple(((k, (val if k == var else v)) for k, v in key_tuple))

    for key, exp in full_exps:
        for qsize in query_sizes:
            new_key = replace(key, 'qsize', qsize)
            if new_key not in experiments:
                new_params = dict(exp.params.items())
                new_params['qsize'] = qsize
                experiments[new_key] = Experiment(
                    new_params, exp.means_data, exp.sterrs_data)

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
        if not experiment.startswith('201') and not experiment.startswith('linear201'):
            continue
        key, params_dict = get_param_vals(experiment)
        chooser, means, sterrs = load_experiment(concat(folder, experiment))
        if 'choosers' in params_dict:
            assert chooser == params_dict['choosers']
        else:
            key = key + (('choosers', chooser),)
            params_dict['choosers'] = chooser
        experiments[key] = Experiment(params_dict, means, sterrs)
    experiments, control_var_vals = simplify_keys(experiments)
    fix_special_cases(experiments)
    changing_vars = [var for var, val in experiments.keys()[0]]
    return experiments, changing_vars, control_var_vals



def get_matching_experiments(experiments, params_to_match):
    """Returns a list of Experiments whose param values match the bindings in
    params_to_match.

    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects
    - params_to_match: Tuple of the form ((var, val), ...) where var is a string
          and val is a string or number. The parameter values to match.
    """
    result = []
    for exp in experiments.values():
        if not all(exp.params[k] == v for k, v in params_to_match):
            continue
        result.append(exp)
    return result

def graph_all(experiments, all_vars, x_var, dependent_vars, independent_vars,
              controls, extra_experiment_params, folder, args):
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
    double_envs = args.double_envs
    if double_envs:
        remaining_vars = list(set(remaining_vars) - set(['mdp', 'beta']))

    extra_experiments = []
    for exp_params in extra_experiment_params:
        identified_experiments = get_matching_experiments(experiments, exp_params)
        # assert len(identified_experiments) == 2
        # assert set([e.params['mdp'] for e in identified_experiments]) == set(['gridworld', 'bandits'])
        extra_experiments.extend(identified_experiments)

    for exp in get_matching_experiments(experiments, controls):
        key = ','.join(['{0}={1}'.format(k, exp.params[k]) for k in remaining_vars])
        if key not in graphs_data:
            graphs_data[key] = extra_experiments[:]  # Make a copy of the list
        graphs_data[key].append(exp)

    for key, exps in graphs_data.items():
        # keys.append(key), graphs_list.append(exps)
        graph(exps, x_var, dependent_vars, independent_vars, controls, key, folder, double_envs)








def graph(exps, x_var, dependent_vars, independent_vars, controls,
          other_vals, folder, double_envs):
    """Creates and saves a single graph.

    Arguments are almost the same as for graph_all.
    - other_vals: String of the form "{var}={val},..." specifying values of
          variables not in x_var, dependent_vars, independent_vars, or
          controls.
    """

    # Whole figure layout setting
    set_style()
    num_rows = len(dependent_vars)
    num_columns = 2 if double_envs else 1
    fig, axes = plt.subplots(num_rows, num_columns, sharex=True)
    sns.set_context(rc={'lines.markeredgewidth': 1.0})   # Thickness or error bars
    capsize = 0.    # length of horizontal line on error bars
    spacing = 100.0

    # Draw all lines and labels
    for row, y_var in enumerate(dependent_vars):
        labels = []
        for experiment in exps:
            col = 0 if experiment.params['mdp'] == 'bandits' else 1
            ax = get_ax(axes, row, num_rows, num_columns, col)

            params = experiment.params
            means, sterrs = experiment.means_data, experiment.sterrs_data
            i_var_val = ', '.join([str(params[k]) for k in independent_vars])
            label = i_var_to_label(i_var_val) + ', '+ str(params['qsize']) # name in legend
            color = chooser_to_color(i_var_val, qsize=params['qsize'])
            x_data = np.array(means[x_var]) + 1
            ax.errorbar(x_data, means[y_var], yerr=sterrs[y_var], color=color,
                         capsize=capsize, capthick=1, label=label)#,
                         # marker='o', markerfacecolor='white', markeredgecolor=chooser_to_color(chooser),
                         # markersize=4)

            labels.append(label)

            ax.set_xlim([0,21])
            ax.set_ylim(-0.15)

            # Set ylabel
            ax_left = get_ax(axes, row, num_rows, num_columns, 0)
            ax_left.set_ylabel(var_to_label(y_var), fontsize=15)

            # Set title
            # title = 'Data for {0}'.format(', '.join(independent_vars))
            title = get_title(col)
            ax_top = get_ax(axes, 0, num_rows, num_columns, col)
            ax_top.set_title(title, fontsize=16, fontweight='normal')


    'Make legend'
    ax = axes[0]
    plt.sca(ax)
    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(handles, labels, range(len(labels))),    # Sorts legend by putting labels 0:k to place as specified
           key=lambda elem: elem[2])
    hl = [[handle, label] for handle, label, idx in hl]
    handles2, labels2 = zip(*hl)

    # ax.legend(handles2, labels2, fontsize=10)
    plt.legend(handles2, labels2, fontsize=11)


    'Change global layout'
    sns.despine(fig)    # Removes top and right graph edges
    plt.suptitle('Number of queries asked', y=0.02, fontsize=16)
    # fig.suptitle('Bandits', y=0.98, fontsize=18)
    # plt.tight_layout(w_pad=0.02, rect=[0, 0.03, 1, 0.95])  # w_pad adds horizontal space between graphs
    # plt.subplots_adjust(top=1, wspace=0.35)     # adds space at the top or bottom
    # plt.subplots_adjust(bottom=.2)
    # fig.set_figwidth(15)     # Can be adjusted by resizing window
    # fig.set_figheight(5)

    'Save file'
    subtitle = ','.join(['{0}={1}'.format(k, v) for k, v in controls])
    subtitle = '{0},{1}'.format(subtitle, other_vals).strip(',')
    folder = concat('graph', folder)
    filename = '{0}-vs-{1}-for-{2}-with-{3}.png'.format(
        ','.join(dependent_vars), x_var, ','.join(independent_vars), subtitle)
    if not os.path.exists(folder):
        os.mkdir(folder)
    # plt.show()
    plt.savefig(concat(folder, filename))


'''TODO:
-Make 2nd set of plots

    -Legend for qsizes
    -Get data for continuous (or at least disc optim with same settings as last experiments)
-List types of graphs I want and make a general framework for creating them
    -2nd set
        -1-2 -i qsize+continuous+discrete optim
            - TODO: Make choosers in [c, d, c+d]
            - TODO: have qsizes seperately for discrete and continuous
    -Compare continuous query sizes among themselves
    -Compare continuous against random continuous for a given qsize
    -Single graphs as usual (but with continuous +discrete optim included)
'''

"""
Plot types
    -4 choosers on bandits and gridworlds for entropy and test regret
        -qsize=5
        -add discrete optim?
    -qsize=2,3,5,10,full,continuous
        -2 or 3 graphs
        -gridworlds + bandits
        -test regret (+entropy? may be boring)
    -continuous: 1,2,3,random baselines,post_avg baselines
    -bar graph for time?
        -qsize=2 or 3
        -random vs greedy vs continuous
"""



def get_ax(axes, row, num_rows, num_columns, col):
    onecol = num_columns == 1
    onerow = num_rows == 1

    # col = 0 if experiment.params['mdp'] == 'bandits' or num_columns == 1 else 1

    if not onerow and not onecol:
        ax = axes[row, col]
    elif onecol and not onerow:
        ax = axes[row]
    elif not onecol and onerow:
        ax = axes[col]
    elif onecol and onerow:
        ax = axes
    else:
        raise ValueError('Number of dependent vars and envs each must be 1 or 2')
    return ax


def get_title(axnum):
    if axnum == 0:
        return 'Bandits*'
    elif axnum == 1:
        return 'Gridworlds'
    else:
        return str(axnum)

def create_legend(ax):
    lines = [
        ('nominal', {'color': '#f79646', 'linestyle': 'solid'}),
        ('risk-averse', {'color': '#f79646', 'linestyle': 'dashed'}),
        ('nominal', {'color': '#cccccc', 'linestyle': 'solid'}),
        ('IRD-augmented', {'color': '#cccccc', 'linestyle': 'dotted'}),
        ('risk-averse', {'color': '#cccccc', 'linestyle': 'dashed'})
    ]

    def create_dummy_line(**kwds):
        return mpl.lines.Line2D([], [], **kwds)

    ax.legend([create_dummy_line(**l[1]) for l in lines],
    [l[0] for l in lines],
    loc='upper right',
    ncol=1,
    fontsize=10,
    bbox_to_anchor=(1.1, 1.0))  # adjust horizontal and vertical legend position



def set_style():
    mpl.rcParams['text.usetex'] = True
    mpl.rc('font', family='serif', serif=['Palatino'])  # Makes font thinner

    sns.set(font='serif', font_scale=1.4)   # Change font size of (sub) title and legend. Serif seems to have no effect.

    # Make the background a dark grid, and specify the
    # specific font family
    sns.set_style("white", {     # Font settings have no effect
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"]})
        # 'axes.facecolor': 'darkgrid'})
        # 'lines.markeredgewidth': 1})


def plot_sig_line(ax, x1, x2, y1, h, padding=0.3):
    '''
    Plots the bracket thing denoting significance in plots. h controls how tall vertically the bracket is.
    Only need one y coordinate (y1) since the bracket is parallel to the x-axis.
    '''
    ax.plot([x1, x1, x2, x2], [y1, y1 + h, y1 + h, y1], linewidth=1, color='k')
    ax.text(0.5*(x1 + x2), y1 + h + padding * h, '*', color='k', fontsize=16, fontweight='normal')


def var_to_label(varname):
    if varname in ['true_entropy']:
        return 'Entropy'
    if varname in ['test_regret']:
        return 'Regret in test env'
    if varname in ['post_regret']:
        return 'Regret in training environment'
    if varname in ['time']:
        return 'Seconds per iteration'
    if varname in ['norm post_avg-true']:
        return 'Distance of posterior average to true reward'

    if varname in ['iteration']:
        return 'Number of queries asked'


def chooser_to_color(chooser, qsize=None):
    greedy_color = 'darkorange' #'lightblue'
    exhaustive_color = 'orange' # 'peachpuff', 'crimson'
    random_color = 'darkgrey' # 'darkorange'
    full_color = 'lightgrey' # 'grey'


    feature_color = 'blue'
    feature_color_random = 'lightblue'
    search_color = 'olivedrab'
    both_color = 'steelblue'

    if chooser.startswith('greedy'):
        if qsize == 2:
            return 'peachpuff'
        if qsize == 3:
            return 'orange'
        if qsize == 5:
            return 'darkorange'
        if qsize == 10:
            return 'orangered'

    if chooser in ['greedy_entropy_discrete_tf', 'greedy_discrete']:
        return greedy_color
    if chooser == 'random':
        return random_color
    if chooser in ['exhaustive', 'exhaustive_entropy']:
        return exhaustive_color
    if chooser == 'full':
        return full_color
    if chooser == 'feature_entropy_init_none':
        return feature_color
    if chooser == 'feature_entropy_random_init_none':
        return feature_color_random
    if chooser == 'feature_entropy_search_then_optim':
        return both_color
    if chooser == 'feature_entropy_search':
        return search_color
    # if chooser == 'Feature selection, random init':
    #     return 'lightorange'
    # if chooser == 'Feature selection, no optimization':
    #     return '=orange'
    # if chooser == 'Feature selection, random init':
    #     return 'darkorange'

def i_var_to_label(i_var):
    if i_var in ['greedy_entropy_discrete_tf', 'greedy_discrete']:
        return 'Greedy'
    if i_var in ['exhaustive', 'exhaustive_entropy']:
        return 'Exhaustive'
    if i_var == 'full':
        return 'Full IRD'
    if i_var == 'random':
        return 'Random'
    if i_var == 'feature_entropy_init_none':
        return 'Feature selection, random init'
    if i_var == 'feature_entropy_random_init_none':
        return 'Feature selection, no optimization, random init'
    if i_var == 'feature_entropy':
        return 'Feature selection'
    if i_var == 'joint_optimize':
        return 'Joint optimize'
    if i_var == 'greedy_optimize':
        return 'greedy_optimize'
    if type(i_var) == int:
        return i_var
    else:
        return i_var

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-x', '--x_var', default='iteration')
    parser.add_argument('-d', '--dependent_var', action='append', required=True)
    parser.add_argument('-i', '--independent_var', action='append', required=True)
    parser.add_argument('-c', '--control_var_val', action='append', default=[])
    parser.add_argument('-e', '--experiment', action='append', default=[])
    parser.add_argument('--double_envs', action='store_true')
    # parser.add_argument('-choosers', '--cho', action='append', default=[])
    return parser.parse_args()

def parse_kv_pairs(lst):
    result = [kv_pair.split('=') for kv_pair in lst]
    return [(k, maybe_num(v)) for k, v in result]

if __name__ == '__main__':
    args = parse_args()
    experiments, all_vars, _ = load_data(args.folder)
    controls = parse_kv_pairs(args.control_var_val)
    extra_experiments = [parse_kv_pairs(x.split(',')) for x in args.experiment]
    graph_all(experiments, all_vars, args.x_var, args.dependent_var,
              args.independent_var, controls, extra_experiments, args.folder, args)
