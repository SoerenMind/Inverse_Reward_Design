
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
import collections

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
    default_values = {}
    for key in keys:
        for name, val in key:
            default_values[name] = val

    flag_name_sets = [set([name for name, _ in key]) for key in keys]
    all_names = set([])
    for name_set in flag_name_sets:
        all_names = all_names.union(name_set)

    def extend_key(key):
        def find_value(flag_name):
            values = [value for name, value in key if name == flag_name]
            return values[0] if values else None
        return tuple([(name, find_value(name)) for name in all_names])

    keys = [extend_key(key) for key in keys]

    indices_with_no_variation = []
    indices_with_variation = []
    for index in range(len(default_values)):
        try:
            if all(key[index][1] == None or key[index][1] == default_values[key[index][0]] for key in keys):
                indices_with_no_variation.append(index)
            else:
                indices_with_variation.append(index)
        except:
            pass
            # indices_with_no_variation.append(index)

    def simple_key(key):
        return tuple((key[index] for index in indices_with_variation))

    new_experiments = {simple_key(extend_key(k)): v for k, v in experiments.items()}
    controls = dict([(keys[0][index][0], default_values[keys[0][index][0]]) for index in indices_with_no_variation])
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
    if len(experiments) < 1:
        raise ValueError('No suitable experiments in folder')
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
    if args.double_envs:
        remaining_vars = list(set(remaining_vars) - set(['mdp', 'beta']))

    extra_experiments = []
    for exp_params in extra_experiment_params:
        identified_experiments = get_matching_experiments(experiments, exp_params)
        # assert len(identified_experiments) == 2
        # assert set([e.params['mdp'] for e in identified_experiments]) == set(['gridworld', 'bandits'])
        extra_experiments.extend(identified_experiments)

    # if args.only_extras:
    #     experiments = {}
    for exp in get_matching_experiments(experiments, controls):
        key = ','.join(['{0}={1}'.format(k, exp.params[k]) for k in remaining_vars])
        if key not in graphs_data:
            graphs_data[key] = extra_experiments[:]  # Make a copy of the list
        if not args.only_extras:
            graphs_data[key].append(exp)

    if x_var != 'qsize':
        for key, exps in graphs_data.items():
            # keys.append(key), graphs_list.append(exps)
            graph(exps, x_var, dependent_vars, independent_vars, controls, key, folder, args)
    else:
        for key, exps in graphs_data.items():
            bar_graph_qsize(exps, x_var, dependent_vars, independent_vars, controls, key, folder, args)

def bar_graph_qsize(exps, x_var, dependent_vars, independent_vars, controls, other_vals, folder, args):
    set_style()
    num_columns = 2 if args.double_envs else 1
    fig, axes = plt.subplots(1, num_columns, sharex=True)
    sns.set_context(rc={'lines.markeredgewidth': 1.0})   # Thickness or error bars
    # capsize = 0.    # length of horizontal line on error bars
    [y_var] = dependent_vars
    x_data = []

    labels = []
    cum_regrets = []

    def params_to_x_pos_and_color_and_label(params):
        qsize = int(params['qsize'])
        if params['choosers'] == 'feature_entropy_search_then_optim':
            pass
        if params['choosers'] == 'greedy_discrete':
            x_pos = [None, None, 0.0, 1., None, 2., None, None, None, None, 3.][qsize]
            color = 'orange'
            label = 'Discrete queries'
        elif params['choosers'] == 'feature_entropy_search_then_optim':
            x_pos = [None, 5., 6., 7.][qsize]
            x_pos +=  - 0.35
            color = 'lightblue'
            label = 'Feature queries'
        else:
            x_pos, color, label = None, None, None
        return x_pos, color, label

    for experiment in exps:
        col = 0 if experiment.params['mdp'] == 'bandits' else 1
        ax = get_ax(axes, 0, 1, num_columns, col)
        params = experiment.params
        if args.exclude:
            if any(str(params[key]) in args.exclude for key in params.keys()):
                continue

        means = experiment.means_data
        num_iter = len(means[y_var])
        cum_regret = means[y_var][num_iter-1]

        x_pos, color, label = params_to_x_pos_and_color_and_label(params)
        if x_pos is not None:
            ax.bar([x_pos], [cum_regret], yerr=[1], color=color, label=label)
            labels.append(label)


        # shift = -0.175
        # plt.xticks([0.0 + shift + 0.5*barwidth, 1.125 + shift + barwidth, 1.125 + shift + 2*barwidth, 1.125 + shift + 3*barwidth,
        #             1.125 + shift + 5*barwidth, 1.125 + shift + 6*barwidth, 1.125 + shift + 7*barwidth],
        #            [2,3,5,10,1,2,3],fontsize=12)
    shift = -0.1
    barwidth = 1.0 + shift
    plt.xticks(
        [0.0 + shift + 0.5 * barwidth, 1. + shift + 0.5 * barwidth, 2. + shift + 0.5 * barwidth,
         3. + shift + 0.5 * barwidth,
         5. + shift + 0.5 * barwidth - 0.3, 6. + shift + 0.5 * barwidth - 0.3, 7. + shift + 0.5 * barwidth - 0.3],
        [2, 3, 5, 10, 1, 2, 3], fontsize=12)

    ax.set_xlim([-0.25, 7.7])
    ax_left = get_ax(axes, 1, 1, num_columns, 0)
    ax_left.set_ylabel(var_to_label(y_var), fontsize=17)
    # Set title
    title = get_title(col)
    ax_top = get_ax(axes, 0, 1, num_columns, col)
    ax_top.set_title(title, fontsize=17, fontweight='normal')


    'Make legend'
    try:
        ax = axes[-1][-1]
    except TypeError:
        try: ax = axes[-1]
        except TypeError:
            ax = axes
    # for ax in flatten(axes):
    plt.sca(ax)
    handles, labels = ax.get_legend_handles_labels()
    # try: legend_order = sorted([int(label) for label in labels])
    legend_order = [2,3,0,1]   # for outperform_IRD
    # legend_order = range(len(labels))   # for discrete
    # legend_order = [1,0,2]  # for continuous
    hl = sorted(zip(handles, labels, legend_order),    # Sorts legend by putting labels 0:k to place as specified
           key=lambda elem: elem[2])
    hl = [[handle, label] for handle, label, idx in hl]
    try:
        handles2, labels2 = zip(*hl)
    except ValueError:
        handles2, labels2 = [], []
        print Warning('Warning: Possibly data only exists for one environment')

    # ax.legend(handles2, labels2, fontsize=10)
    plt.legend(handles2, labels2, fontsize=13)


    'Change global layout'
    sns.despine(fig)    # Removes top and right graph edges
    plt.suptitle('Number of queries asked', y=0.0, x=0.52, fontsize=17, verticalalignment='bottom')
    fig.subplots_adjust(left=0.09, right=.96, top=0.92, bottom=0.12)
    # plt.tight_layout(w_pad=0.02, rect=[0, 0.03, 1, 0.95])  # w_pad adds horizontal space between graphs
    # plt.subplots_adjust(top=1, wspace=0.35)     # adds space at the top or bottom
    # plt.subplots_adjust(bottom=.2)
    fig.set_figwidth(7.5)  # Can be adjusted by resizing window

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
    plt.close()


def graph(exps, x_var, dependent_vars, independent_vars, controls,
          other_vals, folder, args):
    """Creates and saves a single graph.

    Arguments are almost the same as for graph_all.
    - other_vals: String of the form "{var}={val},..." specifying values of
          variables not in x_var, dependent_vars, independent_vars, or
          controls.
    """

    # Whole figure layout setting
    set_style()
    num_rows = len(dependent_vars)
    num_columns = 2 if args.double_envs else 1
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
            if args.exclude:
                if any(str(params[key]) in args.exclude for key in params.keys()):
                    continue
            means, sterrs = experiment.means_data, experiment.sterrs_data
            i_var_val = ', '.join([str(params[k]) for k in independent_vars])
            # if 'full' in i_var_val:
            #     means, sterrs = constant_data_full_IRD(means, sterrs, y_var)
            label = i_var_to_label(i_var_val) + (', '+ str(params['qsize'])) * args.compare_qsizes # name in legend
            color = chooser_to_color(i_var_val, args, params)
            x_data = np.array(means[x_var]) + 1

            try:
                ax.errorbar(x_data, means[y_var], yerr=sterrs[y_var], color=color,
                         capsize=capsize, capthick=1, label=label)#,
                         # marker='o', markerfacecolor='white', markeredgecolor=chooser_to_color(chooser),
                         # markersize=4)
            except:
                pass

            labels.append(label)

            ax.set_xlim([0,21])
            # ylim = ax.get_ylim()
            # ax.set_ylim(ylim)  #-0.15)

            # Set ylabel
            ax_left = get_ax(axes, row, num_rows, num_columns, 0)
            ax_left.set_ylabel(var_to_label(y_var), fontsize=17)

            # Set title
            # title = 'Data for {0}'.format(', '.join(independent_vars))
            title = get_title(col)
            ax_top = get_ax(axes, 0, num_rows, num_columns, col)
            ax_top.set_title(title, fontsize=17, fontweight='normal')


    'Make legend'
    try:
        ax = axes[-1][-1]
    except TypeError:
        try: ax = axes[-1]
        except TypeError:
            ax = axes
    # for ax in flatten(axes):
    plt.sca(ax)
    handles, labels = ax.get_legend_handles_labels()
    # try: legend_order = sorted([int(label) for label in labels])
    legend_order = [2,3,0,1]   # for outperform_IRD
    # legend_order = range(len(labels))   # for discrete
    # legend_order = [1,0,2]  # for continuous
    hl = sorted(zip(handles, labels, legend_order),    # Sorts legend by putting labels 0:k to place as specified
           key=lambda elem: elem[2])
    hl = [[handle, label] for handle, label, idx in hl]
    try:
        handles2, labels2 = zip(*hl)
    except ValueError:
        handles2, labels2 = [], []
        print Warning('Warning: Possibly data only exists for one environment')

    # ax.legend(handles2, labels2, fontsize=10)
    plt.legend(handles2, labels2, fontsize=13)


    'Change global layout'
    sns.despine(fig)    # Removes top and right graph edges
    plt.suptitle('Number of queries asked', y=0.0, x=0.52, fontsize=17, verticalalignment='bottom')
    fig.subplots_adjust(left=0.09, right=.96, top=0.92, bottom=0.12)
    # plt.tight_layout(w_pad=0.02, rect=[0, 0.03, 1, 0.95])  # w_pad adds horizontal space between graphs
    # plt.subplots_adjust(top=1, wspace=0.35)     # adds space at the top or bottom
    # plt.subplots_adjust(bottom=.2)
    fig.set_figwidth(7.5)  # Can be adjusted by resizing window

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
    plt.close()


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el





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
        return 'Shopping'
    elif axnum == 1:
        return 'Chilly World'
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
        return 'Entropy $\mathcal{H}[w^*|\mathcal{D}]$'
    if varname in ['test_regret']:
        return 'Regret in test envs'
    if varname in ['post_regret']:
        return 'Regret in training environment'
    if varname in ['cum_test_regret']:
        return 'Cumulative test regret'
    if varname in ['time']:
        return 'Seconds per iteration'
    if varname in ['norm post_avg-true']:
        return 'Distance of posterior average to true reward'

    if varname in ['iteration']:
        return 'Number of queries asked'
    return varname


def chooser_to_color(chooser, args, params):
    greedy_color = 'darkorange' #'lightblue'
    exhaustive_color = 'crimson' # 'peachpuff', 'crimson'
    random_color = 'darkgrey' # 'darkorange'
    full_color = 'lightgrey' # 'grey'


    # Colors to distinguish optimizers
    # feature_color = 'blue'
    # feature_color_random = 'lightblue'
    # search_color = 'olivedrab'
    # both_color = 'steelblue'

    # Colors to only distinguish optimized vs not optimized
    optimized_color = 'lightblue'#0066FF' # blue
    feature_color = optimized_color
    feature_color_random = 'forestgreen'
    search_color = optimized_color
    both_color = optimized_color

    # Different colors per qsize if comparing qsizes
    if args.compare_qsizes:
        qsize = params['qsize']
        if chooser.startswith('greedy'):
            if qsize == 2:
                return '#00CCFF'
            if qsize == 3:
                return '#0066FF'
            if qsize == 5:
                return '#0000FF'
            if qsize == 10:
                return '#000099'
        elif chooser.startswith('feature_entropy'):
            if qsize == 1:
                return '#FF6666'
            if qsize == 2:
                return '#FF0000'
            if qsize == 3:
                return '#CC0000'

    # Shades of blue for subsample sizes
    if 'num_subsamp' in args.independent_var:
        subsamp = params['num_subsamp']
        color_dict = {}
        color_dict[2] = '#66FFFF'
        color_dict[5] = '#00CCFF'
        color_dict[10] = '#00CCFF'
        # color_dict[10] = '#0066FF'
        color_dict[50] = '#0033CC'
        color_dict[100] = '#000099'
        color_dict[10000] = 'grey'
        # color_dict[10000] = '#000033'
        return color_dict[subsamp]

    # chooser = chooser.split(",")

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
    if chooser == 'feature_random':
        return random_color
    # if chooser == 'Feature selection, random init':
    #     return 'lightorange'
    # if chooser == 'Feature selection, no optimization':
    #     return '=orange'
    # if chooser == 'Feature selection, random init':
    #     return 'darkorange'

def i_var_to_label(i_var):

    # Discrete choosers
    if i_var in ['greedy_entropy_discrete_tf', 'greedy_discrete']:
        return 'Greedy'
    if i_var in ['exhaustive', 'exhaustive_entropy']:
        return 'Large search'
    if i_var == 'full':
        return 'IRD'
    if i_var == 'random':
        return 'Random'
    if i_var == 'joint_optimize':
        return 'Joint optimize'
    if i_var == 'greedy_optimize':
        return 'greedy_optimize'
    if type(i_var) == int:
        return i_var

    # Continuous choosers
    if i_var == 'feature_entropy_init_none':
        return 'Features and weights optimized (GD)'
    if i_var == 'feature_entropy_random_init_none':
        return 'Only feature optimized'
    if i_var == 'feature_entropy_search':
        return 'Features and weights optimized (search)'
    if i_var == 'feature_entropy_search_then_optim':
        return 'Weights optimized too'
    if i_var == 'feature_random':
        return 'Unoptimized'

    if i_var == '10000':
        # return 'Repeated full IRD (10000)'
        return '10000 (repeated full IRD)'
    else:
        return i_var

def constant_data_full_IRD(means, sterrs, y_var):
    full_mean, full_std, num_iter = means[y_var][1], sterrs[y_var][1], len(means[y_var])
    means[y_var] = [full_mean for _ in range(num_iter)]
    sterrs[y_var] = [0 for _ in range(num_iter)]
    means[y_var][10], sterrs[y_var][10] = full_mean, full_std
    return means, sterrs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-x', '--x_var', default='iteration')
    parser.add_argument('-d', '--dependent_var', action='append', required=True)
    parser.add_argument('-i', '--independent_var', action='append', required=True)
    parser.add_argument('-c', '--control_var_val', action='append', default=[])
    parser.add_argument('-e', '--experiment', action='append', default=[])
    parser.add_argument('--double_envs', action='store_true')
    parser.add_argument('--compare_qsizes', action='store_true')
    parser.add_argument('--exclude', action='append')
    parser.add_argument('--only_extras', action='store_true')
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