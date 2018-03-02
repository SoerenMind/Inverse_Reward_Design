from subprocess import call

# Discrete experiments
NUM_EXPERIMENTS = '2'  # Modify this to change the sample size

choosers = ['random', 'greedy_entropy_discrete_tf', 'exhaustive_entropy']
query_sizes = ['2', '3', '5', '10']
mdp_types = ['bandits', 'gridworld']

def run(chooser, qsize, mdp_type):
    if mdp_type == 'bandits':
        # Values range from -5 to 5 approximately, so setting beta to 1 makes
        # the worst Q-value e^10 times less likely than the best one
        beta = '1.0'
        beta_planner = '10'
        dim = '10'
        # TODO: Set the following to the right values
        lr = '0.1'
        num_iters_optim = '10'
    else:
        # Values range from 50-100 when using 25 value iterations.
        beta = '0.1'
        beta_planner = '10'
        dim = '10'
        # TODO: Set the following to the right values
        lr = '0.1'
        num_iters_optim = '10'

    call(['python', 'run_IRD.py',
          '-c', chooser,
          '--query_size', qsize,
          '--num_experiments', NUM_EXPERIMENTS,
          '--num_iter', '20',
          '--gamma', '1.0',
          '--size_true_space', '10000',
          '--size_proxy_space', '100',
          '--seed', '1',
          '--beta', beta,
          '--beta_planner', beta_planner,
          '--num_states', '100',  # Only applies for bandits
          '--dist_scale', '0.5',
          '--num_traject', '1',
          '--num_queries_max', '500',
          '--height', '12',  # Only applies for gridworld
          '--width', '12',   # Only applies for gridworld
          '--lr', lr,  # Doesn't matter, only applies in continuous case
          '--value_iters', '25',
          '--mdp_type', mdp_type,
          '--feature_dim', dim,
          '--num_test_envs', '10',
          '--true_rw_random'])

if __name__ == '__main__':
    for mdp_type in mdp_types:
        run('full', '2', mdp_type)

        for chooser in choosers:
            for qsize in query_sizes:
                run(chooser, qsize, mdp_type)
