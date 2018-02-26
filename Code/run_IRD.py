import time

print('importing')

start = time.clock()
import numpy as np
from inference_class import InferenceDiscrete
from gridworld import GridworldEnvironment, Direction, NStateMdpHardcodedFeatures, NStateMdpGaussianFeatures,\
    NStateMdpRandomGaussianFeatures, GridworldMdpWithDistanceFeatures, GridworldMdp
from agents import ImmediateRewardAgent, DirectionalAgent, OptimalAgent
from query_chooser_class import Query_Chooser_Subclass, Experiment
# from interface_discrete import Interface
from random import choice, seed
# from scipy.special import comb
import copy
from utils import Distribution
# from scipy.misc import logsumexp
import sys
import argparse
import tensorflow as tf


print 'Time to import: {deltat}'.format(deltat=time.clock() - start)



def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def pprint(y):
    print(y)
    return y

def choose_regret_minimizing_proposal(set_of_proposal_sets, reward_space_true, prior, inference, cost_of_asking = 0.01):
    '''Chooses a proposal reward sub space by minimizing the expected regret after asking.'''
    """Todo:
    -Functions for looping through sets, exp_post_regret(proxy, omega), get_post_avg(posterior), get_posterior(proxy, omega)
    """
    prior_avg = sum([prior[tuple(reward)] * reward for reward in reward_space_true]) # Over whole reward_space (change?)
    best_r_set = []
    best_exp_exp_post_regret = np.inf
    best_regret_plus_cost = np.inf
    set_of_proposal_sets = list(set_of_proposal_sets)[::-1]
    for omega in set_of_proposal_sets:
        # TODO: Restricted to length 2!
        if not len(omega)==2: continue
        # if not omega == list(set_of_proposal_sets)[2]: continue

        cost_omega = cost_of_asking*len(omega)
        prior_regret = 0
        exp_exp_post_regret = 0
        for proxy in omega:
            # TODO: Extremely costly to get posterior for all proxy choices. Save repeated computations?
            # Do I have to do (and thus save) the planning for every proxy here?
            inference.calc_and_save_posterior(omega, proxy)    # Do only once per reward
            posterior = dict([(tuple(true_reward), inference.get_posterior(true_reward, omega, proxy))
                              for true_reward in reward_space_true])
            post_avg = sum([posterior[tuple(reward)] * reward for reward in reward_space_true]) # Over whole reward_space
            # Calculate expected regret from optimizing post_avg (expectation over posterior true rewards)
            exp_post_regret = sum([posterior[tuple(true_reward)]  # multiply by regret
                     * (inference.get_avg_reward(true_reward,true_reward) - inference.get_avg_reward(post_avg,true_reward)) for true_reward in reward_space_true])
            # sum_true posterior(true | prox)* (avg_reward(true | true) - avg_reward(proxy | true))
            exp_exp_post_regret += exp_post_regret
        regret_plus_cost = exp_exp_post_regret + cost_omega
        if regret_plus_cost < best_regret_plus_cost:
            best_regret_plus_cost = regret_plus_cost
            best_exp_exp_post_regret = exp_exp_post_regret
            best_r_set = omega
            best_posterior = posterior
            best_post_avg = post_avg
    return best_r_set, best_exp_exp_post_regret, best_regret_plus_cost, best_posterior, best_post_avg

    query_chooser = Query_Chooser_Subclass(inference, reward_space_proxy, cost_of_asking=0.)
    set_of_queries = list(query_chooser.generate_set_of_queries(query_size))
    print(len(set_of_queries))
    best_query, best_regret, _ = query_chooser.find_regret_minimizing_query(set_of_queries)

# # @profile
def experiment(inference_sim, reward_space_proxy, query_size, num_queries_max, iterations_random=10,
               iterations_optimized=20, greedy=True):
    exp_regret_diff = []
    exp_regret_gain = []
    regret_compare = []
    regret_exp_vs_actual = []
    inference_sim = copy.deepcopy(inference_sim)
    for i in range(iterations_optimized):
        print('Experiment number:{i}/{iter}'.format(i=i,iter=iterations_optimized))

        # Set up query chooser and inference
        inference_sim.agent.mdp.populate_features()
        inference_sim.feature_expectations_dict = {}    # Replace these lines with inference.reset()
        query_chooser = Query_Chooser_Subclass(inference_sim, reward_space_proxy, cost_of_asking=0.)
        set_of_queries = list(query_chooser.generate_set_of_queries(query_size, max_num_queries=1000))
        random_query = choice(set_of_queries)

        # Select query and record expected regrets
        if greedy == True:
            best_query, best_regret, _     = query_chooser.find_best_query_greedy(query_size)
        elif greedy == False:
            best_query, best_regret, _     = query_chooser.find_regret_minimizing_query(set_of_queries)
        elif greedy == 'maxmin':
            best_query, best_regret, _ = query_chooser.find_query_feature_diff(query_size)
        _, random_regret, _ = query_chooser.find_regret_minimizing_query([random_query]) # Sometimes finds lower regret for empty query
        _, prior_regret, _             = query_chooser.find_regret_minimizing_query([])
        exp_regret_diff.append(random_regret - best_regret) # This should match the actual regret diff on average
        exp_regret_gain.append(prior_regret - best_regret)

        # Do inference on chosen query, compare regret to that of random query and prior regret
        regret_optimized, optimized_actual_std = get_regret_from_query(inference_sim, best_query)
        regret_random_query, random_actual_std = get_regret_from_query(inference_sim, random_query)
        print optimized_actual_std
        print random_actual_std
        regret_compare.append((regret_optimized, regret_random_query))
        regret_exp_vs_actual.append((best_regret, regret_optimized))

    regret_diff_actual = np.array([x-y for x,y in regret_compare])
    mean_std_regret_diff_actual = (regret_diff_actual.mean(), regret_diff_actual.std())
    return np.array(exp_regret_diff).mean(), np.array(exp_regret_diff).std(), sum(sum([np.array(exp_regret_diff) < 0])), \
           np.array(exp_regret_gain).mean(), mean_std_regret_diff_actual, regret_compare, regret_exp_vs_actual


def get_regret_from_query(inference_eval, best_query, num_true_rewards=500):
    regrets = []
    # for j in range(num_true_rewards):
    #     true_reward = choice(reward_space_true)  # Replace with sample from
    for true_reward in reward_space_true:
        lhoods = []
        for i, proxy in enumerate(best_query):
            lhood = inference_eval.get_likelihood(true_reward, best_query, proxy)
            lhoods.append(lhood)
        # chosen_proxy_number = np.array(lhoods).argmax()  # Replace argmax with sampling
        d = {i: lhood for i, lhood in enumerate(lhoods)}
        try: chosen_proxy_number = Distribution(d).sample()
        except:
            chosen_proxy_number = np.array(lhoods).argmax()  # Replace argmax with sampling
        chosen_proxy = best_query[chosen_proxy_number]
        inference_eval.calc_and_save_posterior(best_query, chosen_proxy)
        post_avg = inference_eval.get_posterior_avg(best_query, chosen_proxy)
        # TODO: Make a query_chooser / inference.function from query to regret or so
        optimal_reward = inference_eval.get_avg_reward(true_reward, true_reward)
        post_reward = inference_eval.get_avg_reward(post_avg, true_reward)
        regret = optimal_reward - post_reward
        regrets.append(regret)
    avg_regret = np.array(regrets).mean()
    std_regret = np.array(regrets).std()
    return avg_regret, std_regret


def test_planning_speed(inference, reward_space_proxy):
    print('testing planning speed')
    for i, proxy in enumerate(reward_space_proxy):
        inference.get_feature_expectations(proxy)

# ==================================================================================================== #
# ==================================================================================================== #
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('q_size',type=int)
    # parser.add_argument('choosers',type=list,default='[greedy_entropy]')
    parser.add_argument('-c','--c', action='append', required=True) # c for choosers
    parser.add_argument('--query_size_feature',type=int,default=3)
    parser.add_argument('--num_experiments',type=int,default=1)
    parser.add_argument('--num_iter',type=int,default=5)
    # TODO: Values are computed as if trajectories are infinite. Problem?
    parser.add_argument('--gamma',type=float,default=0.95)
    parser.add_argument('--size_true_space',type=int,default=200)
    parser.add_argument('--size_proxy_space',type=int,default=50)
    parser.add_argument('--num_trajectories',type=int,default=1)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--beta',type=float,default=2.)
    parser.add_argument('--num_states',type=int,default=6)
    parser.add_argument('--dist_scale',type=float,default=0.5)
    parser.add_argument('--num_traject',type=int,default=1)
    parser.add_argument('--num_queries_max',type=int,default=500)
    parser.add_argument('--height',type=int,default=8)
    parser.add_argument('--width',type=int,default=8)
    parser.add_argument('--num_iters_optim',type=int,default=5)
    parser.add_argument('--value_iters',type=int,default=25)
    parser.add_argument('--mdp_type',type=str,default='gridworld')


    args = parser.parse_args()


    # Experiment description
    adapted_description = False
    print "Adapted description: ", adapted_description
    exp_description = pprint("Comparing to entropy with many states and few true rewards. {nexp} experiments.")

    # Set parameters
    dummy_rewards = np.zeros(3)
    # TODO: Randomize goal positions per experiment
    goals = [(1,1), (2,6), (3,3), (3,4), (4,5), (6,4), (6,6)]
    # goals = [(1,1), (2,6), (3,3)]
    # feature_dim = args.feature_dim
    args.feature_dim = len(goals)   # Overwriting arg input
    # Set parameters
    choosers = args.c
    SEED = args.seed
    seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    beta = args.beta
    num_states = args.num_states
    size_reward_space_true = args.size_true_space
    size_reward_space_proxy = args.size_proxy_space
    num_queries_max = args.num_queries_max
    num_traject = args.num_traject
    num_experiments = args.num_experiments
    num_iter_per_experiment = args.num_iter #; print('num iter = {i}'.format(i=num_iter_per_experiment))
    # Params for Gridworld
    gamma = args.gamma
    query_size = args.q_size
    dist_scale = args.dist_scale
    height = args.height
    width = args.width
    num_iters_optim = args.num_iters_optim
    proxy_subspace = True
    # choosers = ['greedy', 'greedy_exp_reward']
    # choosers = ['no_query','greedy_entropy', 'greedy', 'greedy_exp_reward', 'random']
    # choosers = ['greedy_entropy', 'random', 'no_query']

    exp_params = 'qsize'+str(query_size) + '-' + 'expnum' + str(num_experiments)
    exp_name = 'compare-choosers'


    # # Set up env and agent for NStateMdp
    if args.mdp_type == 'bandits':
        # mdp = NStateMdpRandomGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
        #                                 feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED)
        mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=np.zeros(args.feature_dim), start_state=0, preterminal_states=[],
                                        feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED)
        agent = ImmediateRewardAgent()

        # Reward spaces for N-State-Mdp
        from itertools import product
        reward_space_true = list(product([0,1], repeat=args.feature_dim))
        # reward_space_true.remove((0,0,0,0))
        # TODO(rohinmshah): These reward spaces have many copies of the same reward function
        reward_space_true = [np.array(reward) for reward in reward_space_true]
        reward_space_true = [choice(reward_space_true) for _ in range(size_reward_space_true)]
        # reward_space_true = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0])]
        reward_space_proxy = [choice(reward_space_true) for _ in range(size_reward_space_proxy)]
        # reward_space_proxy = reward_space_true
        # len_reward_space = len(reward_space_true)
        # reward_space = [np.array([1,0]),np.array([0,1]), np.array([1,1])]

    # Set up env and agent for gridworld
    elif args.mdp_type == 'gridworld':
        grid = GridworldMdp.generate_random(height,width,0.1,0.2,goals,living_reward=-0.01, print_grid=True)
        mdp = GridworldMdpWithDistanceFeatures(grid, dist_scale, living_reward=-0.01, noise=0, rewards=dummy_rewards)
        agent = OptimalAgent(gamma, num_iters=args.value_iters)



        # Create reward spaces for gridworld
        # reward_space_true = [np.random.multinomial(18, np.ones(args.feature_dim)/18) for _ in xrange(size_reward_space_true)]
        # reward_space_proxy = [np.random.multinomial(18, np.ones(args.feature_dim)) for _ in xrange(size_reward_space_proxy)]
        reward_space_true = [np.random.randint(-9, 9, size=[args.feature_dim])   for _ in xrange(size_reward_space_true)]
        if proxy_subspace:
            reward_space_proxy = [choice(reward_space_true) for _ in xrange(size_reward_space_proxy)]
        else:
            reward_space_proxy = [np.random.randint(-9, 9, size=[args.feature_dim])   for _ in xrange(size_reward_space_proxy)]
        # reward_space_true = [np.random.dirichlet(np.ones(args.feature_dim)) * args.feature_dim - 1 for _ in xrange(size_reward_space_true)]
        # reward_space_proxy = [np.random.dirichlet(np.ones(args.feature_dim)) * args.feature_dim - 1 for _ in xrange(size_reward_space_proxy)]

    else:
        raise ValueError('Unknown MDP type: ' + str(args.mdp_type))



    # Set up inference
    env = GridworldEnvironment(mdp)
    inference = InferenceDiscrete(
        agent, mdp, env, beta, reward_space_true, reward_space_proxy,
        num_traject=num_traject, prior=None)

    'Print derived parameters'
    # print('Size of reward_space_true:{size}'.format(size=size_reward_space_true))
    # print('Size of reward_space_proxy:{size}'.format(size=len(reward_space_proxy)))
    # print('Query size:{size}'.format(size=query_size))
    # print('Choosers: {c}').format(c=choosers)
    # if greedy == False:
    #     num_queries = min([comb(len(reward_space_proxy), query_size),    num_queries_max])
    #     # num_queries = comb(len(reward_space_proxy), query_size)
    #     num_post_avg_plans = num_queries * query_size
    # else:
    #     num_queries = (query_size-1) * len(reward_space_proxy)
    #     avg_query_size = (query_size+2)/2.
    #     num_post_avg_plans = num_queries * avg_query_size
    # print('Number of queries: min({size},{max})'.format(size=num_queries,max=num_queries_max))
    # num_planning_problems = len(reward_space_proxy) + num_post_avg_plans + size_reward_space_true
    # print('Number of rewards to plan with:{size}'.format(size=num_planning_problems))
    # print('Greedy: {g}').format(g=greedy)
    print('======================================================================================================')



    'Set up test environment (not used)'
    # print 'starting posterior calculation'
    # inference.calc_and_save_posterior(reward_space_proxy, proxy_given)
    # prior = dict([(tuple(true_reward), inference.get_posterior(true_reward, reward_space_proxy, proxy_given))
    #               for true_reward in reward_space_true])
    # print('new prior: {prior}'.format(prior=prior))
    # mdp_test = NStateMdpGaussianFeatures(num_states=num_states, rewards=proxy_given, start_state=0, preterminal_states=[],   # proxy_given should have no effect
    #                                      feature_dim=args.feature_dim, num_states_reachable=num_states, SEED=SEED)
    # mdp_test.add_feature_map(mdp.features)
    # env_test = GridworldEnvironment(mdp_test)
    # inference_test = InferenceDiscrete(agent, mdp_test, env_test, beta=1., reward_space_true=reward_space_true, num_traject=1, prior=prior)




    'Experiment'
    def experiment(query_size):
        # test_planning_speed(inference, reward_space_proxy); print('tested planning speed')
        # mean, std, failures, gain, mean_std_actual, regret_compare, regret_exp_vs_actual \
        #     = experiment(inference, reward_space_proxy, query_size, iterations_optimized=num_experiments, greedy=greedy,
        #                  num_queries_max=num_queries_max)
        # print mean, std, failures, gain, mean_std_actual
        # print 'Expected regret improvement over random query (mean): {r}'.format(r=mean)
        # print 'Expected regret improvement over no query (mean): {r}'.format(r=gain)
        # print 'Mean actual -reduction and std(mean) over random query: {r}'.format(r=mean_std_actual)
        # print 'Actual regret diff optimized vs random:{r}'.format(r=regret_compare)
        # print 'Expected vs actual regret: {vs}'.format(vs=regret_exp_vs_actual)
        experiment = Experiment(inference, reward_space_proxy, query_size, num_queries_max, args, choosers, SEED, exp_params, exp_name)
        # experiment.run_experiment(num_iter_per_experiment)
        avg_post_exp_regrets, avg_post_regrets, \
        std_post_exp_regrets, std_post_regrets, \
        results = experiment.get_experiment_stats(num_iter_per_experiment, num_experiments)


        'Print results'
        print "Choosers:                        {c}".format(c=choosers)
        print "Avg post exp regret per chooser: {x}".format(x=avg_post_exp_regrets)
        print "Avg post regret per chooser: {x}".format(x=avg_post_regrets)
        print "Std post exp regret per chooser: {x}".format(x=std_post_exp_regrets)
        print "Std post regret per chooser: {x}".format(x=std_post_regrets)
        # print [-results['greedy_exp_reward','perf_measure',4, n] for n in range(num_experiments)]
        # print [results['greedy_exp_reward','post_exp_regret',4, n] for n in range(num_experiments)]
        print [results['greedy_entropy','post_exp_regret',num_iter_per_experiment-1, n] for n in range(num_experiments)]
        print [results['greedy_entropy','perf_measure',num_iter_per_experiment-1, n] for n in range(num_experiments)]
        # print [results['greedy','perf_measure',4, n] for n in range(num_experiments)]

        print "Entropy per iteration for greedy_entropy:"
        print [np.array([results['greedy_entropy','perf_measure',i, n] for n in range(num_experiments)]).mean() for i in range(num_iter_per_experiment)]
        # print "Entropy per iteration for greedy:"
        # print [[results['greedy','post_entropy',i, n] for n in range(num_experiments)] for i in range(num_iter_per_experiment)]

        # print("Test environment regret for greedy:")
        # print [results['greedy','test_regret',num_iter_per_experiment-1,n] for n in range(num_experiments)]
        print("Test environment regret for greedy_entropy:")
        print [results['greedy_entropy','test_regret',num_iter_per_experiment-1,n] for n in range(num_experiments)]

        # print "Exp regret per iteration for greedy:"
        # print [[results['greedy','post_exp_regret',i, n] for n in range(num_experiments)] for i in range(num_iter_per_experiment)]
        print "Exp regret per iteration for greedy_entropy:"
        print [np.array([results['greedy_entropy','post_exp_regret',i, n] for n in range(num_experiments)]).mean() for i in range(num_iter_per_experiment)]


        # print 'mdp_features:'
        # print np.array([np.concatenate([[state], mdp.features[state]]) for state in range(num_states)])
        # print np.array([np.concatenate([[state], mdp.get_features(state)]) for state in range(num_states)])
        # # # print('mdp features: {features}'.format(features=mdp_test.features))
        # print(best_query)
        # # print('Best post_avg:{post_avg}').format(post_avg=best_post_avg)
        # # print('Best posterior:{posterior}').format(posterior=best_posterior)
        print 'Total time:{deltat}'.format(deltat=time.clock() - start)
        print 'Finished experiment: ', exp_description.format(nexp=num_experiments), adapted_description

    # for q_size in range(2,50):
    #     if q_size % 4 == 0:
    #         experiment(q_size)
    experiment(query_size)


    'Create interface'
    # omega = [choice(reward_space_true) for _ in range(4)] # replace with chosen omega
    # interface = Interface(omega, agent, env, num_states=num_states)
    # interface.plot()

    """Todo:
        -Implement new choosing methods
            -Combineable with sliders / interface
                -Feature weights
                -

            -Entropy

                -Non-greedy entropy - cheap!
                -Why don't other methods reduce entropy? Focus on regret.
                    -Do they?
                -Results:
                    -Random:0.6, regret/reward: ca 0.2, entropy: 0.36
                    -Hard mode: Entropy: .86, regret: .5, reward: .73, random:. 1.6 (5 iter, 5 exp)
                -Try on hard problem. Maybe entropy tries too hard to distinguish between weights that lead to the same behavior once other options are sufficiently unlikely (p=0.01
                    -More states to make same behavior unlikely
                        -Hypothesis: Entropy works better to reduce the search space, regret better to exclude remaining choices that are merely somewhat unlikely.
                    -Optimize with regret at the end?
                        -Check if post_regret drops equally fast initially for entropy
                    -
                -Compare to only H(Q) or H(Q|W)
        -New environment:
            -Desiderata:
            -Features = distance from different (weighted) locations
                -Incentivize a path rather than directly finding location (sq distance?)
            -Ask Rohin/Daniel
            -Dorsa: Too slow to plan?
            -Racecar: Same?
            -With test env...
            -Distance to goals gridworld
                -Run a test, e.g. with Value iteration (first by caching reward and not using features)

        -List parameters and sensitivity to them (remember: messed up last time!)
        -Test new environment
            -Initial test

                -Test final post avg on different environment
                -Why is entropy increasing?
                    -Try bigger proxy space
                        -Result: Helps to have >50-60 but high variance.
                    -Try uniform weights (not right-skewed)
                        -Beta = 2.
                        -Result: Entropy goes 7, then 14 if dividing by 9! Repeats with different seed.
                            -0/-1 rewards due to int division. So trajectories are similar.
                                -But why the huge entropy?
                            -Check w/ float division
                                -Familiar slow decline of ent
                            -WHICH BETA IS REASONABLE?
                                -Check probabilities
                        -Result: Entropy way down when not dividing by 9!

                    -What does the (small) posterior look like?

                    -!!!!!!!!!!!!!! WHY NEGATIVE REGRET?
                        -subspace and convergence for VI don't help
                        -find out what feature_exp for proxy are better than for true reward
                    -Why does entropy go back up sometimes? Maybe bad proxy choice.
                    -Why tiny differences in entropy?
                        -INCREASE BETA! - Check effect for really high in computations
                        -Compare avg_reward_matrix for NStateMdp
                        -There are so many true rewards that maybe a pairwise query
                            -Try exhaustive search (cheap for entropy!)
                        -Maybe a slight change in behavior changes feature exp a lot?
                -Adapt reward space
                    -Constrain vector L1-norm by sampling multinomial sum vector, or Dir(1,1,1,1,1) (may slow planning).
                        -Not the same as independent draws but it's about the ratios.

                -Think about 'discrete' measure from Dorsa paper
                -Change map between experiments
            -Notes:
                -Query mostly just tells you which of the 4 rewards leads to the true final goal. That's easy to
                replicate with an algorithm that asks about final goals. Decrease dist_scale, increase #goals & size to
                make it about path selection.

        -Idea: Use queries to reduce entropy to perform do well in test environment.
            -Keep asking queries which won't improve our optimal behavior in the training env but might in test.
                -Maybe entropy performs worse because it wastes time on this!
                -This is kind of like the 'lookahead' provided by entropy over classification error in RF.
                -Idea: Try getting regret from max likelihood r_true to save time. Or weighted feature exp from top r_true's.
            -Is this the right approach for that problem? Is this an interesting problem?
            -Could also do multiple iterations through multiple environments for transfer or lifelong reward-learning.
                -Pro: Better generalization. Like repeated IRL paper.

        -TODOs in run_experiment (finish posterior vectorization)
            -Replace old methods completely (making inference much smaller)
        -ssh run the whole thing from now on

        -Measure: Total planning vs belief updating
            ==> Both ca 1.5s with quick planning, greedy only, 500/50 spaces
        -Implement greedy with quadratic and compare
        -Efficiency: Vectorize chooser all the way across queries
        -Note: Don't delete old posteriors if needed again (across choosers)
        -Random features
        - Compare minimizer against random query
        -Measure effect of adjusting parameters


    -Try not sampling actions - Change back!
    -Implement Race car / Dorsa domain with between-track generalization
    """
