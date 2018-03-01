from subprocess import call

# Discrete experiments
choosers = ['random', 'greedy_entropy', 'exhaustive_entropy']
query_sizes = ['2', '3', '5', '10']
mdp_types = ['bandits', 'gridworld']

def run(chooser, qsize, mdp_type):
    call(['python', 'run_IRD.py', '-c', chooser, '--mdp_type', mdp_type, qsize])

if __name__ == '__main__':
    for mdp_type in mdp_types:
        run('full_query', '2', mdp_type)

        for chooser in choosers:
            for qsize in query_sizes:
                run(chooser, qsize, mdp_type)
