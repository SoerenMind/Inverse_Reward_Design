from subprocess import call

# Discrete experiments
NUM_EXPERIMENTS = '5'  # Modify this to change the sample size

choosers = ['greedy_entropy_discrete_tf', 'random', 'exhaustive_entropy']
query_sizes = ['2', '3', '5', '10']
mdp_types = ['gridworld', 'bandits']
true_reward_space_sizes = ['1000000']
viters = ['15'] #, '25']

def run(chooser, qsize, mdp_type, discretization_size='5', viter='15', rsize='1000000', subsampling='1'):
    if mdp_type == 'bandits':
        # Values range from -5 to 5 approximately, so setting beta to 1 makes
        # the worst Q-value e^10 times less likely than the best one
        beta = '0.5'
        beta_planner = '10'
        dim = '20'                                                              # Increase to 20?
        # TODO: Set the following to the right values
        lr = '0.01'
        num_iters_optim = '10'
    else:
        # Values range from 50-100 when using 25 value iterations.
        beta = '0.2'                                                            # Increased for linear features
        beta_planner = '10'
        dim = '20'                                                              # Increase to 20?
        # TODO: Set the following to the right values
        lr = '0.1'
        num_iters_optim = '10'

    call(['python', 'run_IRD.py',
          '-c', chooser,
          '--query_size', qsize,
          '--num_experiments', NUM_EXPERIMENTS,
          '--num_iter', '20',
          '--gamma', '1.',
          '--size_true_space', rsize,
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
          '--lr', lr,   # Doesn't matter, only applies in continuous case
          '--value_iters', viter,  # Consider decreasing viters to 10-15 to make the path more important as opposed to ending up at the right goal
          '--mdp_type', mdp_type,
          '--feature_dim', dim,
          '--num_test_envs', '100',
          '--subsampling', subsampling,
          '--num_subsamples','10000',
          '--weighting', '1',
          '--well_spec', '1',
          '--linear_features', '1'
        ])




# Run as usual
def run_discrete():
    for mdp_type in mdp_types:
        run('full', '2', mdp_type)

        for chooser in choosers:
            for qsize in query_sizes:
                run(chooser, qsize, mdp_type)

# Run with different rsize and subsampling values
def run_subsampling():
    for viter in viters:
        for mdp_type in mdp_types:
            for rsize in true_reward_space_sizes:
                if rsize == '10000':
                    subsampling = '0'
                else: subsampling = '1'


                for chooser in choosers:
                    for qsize in query_sizes:
                        run(chooser, qsize, mdp_type, viter, rsize, subsampling)

                run('full', '2', mdp_type, viter, rsize, subsampling)

def run_discrete_optimization():
    for mdp_type in mdp_types:
        for qsize in query_sizes:
            for chooser in ['incremental_optimize', 'joint_optimize']:
                run(chooser, qsize, mdp_type)

def run_continuous():
    for mdp_type in mdp_types:
        for qsize, discretization_size in [('1', '9'), ('2', '5'), ('3', '3')]:
            run('feature_entropy', qsize, mdp_type, discretization_size=discretization_size)

if __name__ == '__main__':
    run_discrete_optimization()
