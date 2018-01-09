import numpy as np
import environment
import agent_class
from inference_class import Inference, test_inference
import itertools
from gridworld import NStateMdp, GridworldEnvironment, Direction, NStateMdpHardcodedFeatures, NStateMdpGaussianFeatures
from agents import ImmediateRewardAgent, DirectionalAgent
from agent_runner import run_agent
# from scipy.stats import itemfreq
from interface_discrete import Interface




def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1, 1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

    '''Regret minimization:
    # -Get regret before asking: sum_r p(r) regret(r_avg, r): wrong?
    #     -Problem: Gotta loop through ALL rewards r and get p(r)*regret(post_avg | r)
    #     -This is gonna be the same for every reward set, so omit! (Unless you assume the set contains the truth)
    -Regret = test_inference.get_avg_reward(proxy, true_reward) - (true | true)
    -Regret after asking:
        for r in omega:
            posterior|r =
            post_avg =
            exp_reward_after_answer = sum_{rs_true} post(r_true) * regret(r_true | post_avg)
            exp_reward_after_asking += prior(r) * exp_reward_after_answer

    '''


# def choose_proposal_delta(reward_space, prior, inference, cost_of_asking = 0.01):
#     '''Chooses a proposal reward sub space by assuming that the picked reward function is the true one (delta posterior)
#     and minimizing regret. Possible issue: calculating things over the proposal set vs the whole reward_space.'''
#     for omega in powerset(reward_space):
#         post_avg = sum([prior[tuple(reward)] * reward for reward in omega])
#         avg_weighted_reward = inference.get_avg_reward(post_avg, post_avg)
#         exp_reward_after_asking = sum([prior[tuple(reward)] * inference.get_avg_reward(reward, reward)
#                                        for reward in omega])
#         gain_for_asking = exp_reward_after_asking - avg_weighted_reward - cost_of_asking*len(omega)
#         if gain_for_asking > best_r_set_gain:
#             best_r_set = omega; best_r_set_gain = gain_for_asking
#     return best_r_set, best_r_set_gain


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
            inference.calc_and_save_posterior(proxy, reward_space_proxy=omega)    # Do only once per reward
            posterior = dict([(tuple(true_reward), inference.get_posterior(true_reward))
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



if __name__=='__main__':
    # Define environment and agent
    SEED = 1
    beta = 100.
    num_states = 6
    feature_dim = 4
    proxy_given = np.array([0,0,1,0])
    mdp = NStateMdpGaussianFeatures(num_states=num_states, rewards=proxy_given, start_state=0, preterminal_states=[],
                                    feature_dim=feature_dim, num_states_reachable=num_states, SEED=SEED)
    env = GridworldEnvironment(mdp)
    agent = ImmediateRewardAgent()
    agent.set_mdp(mdp)
    # print(run_agent(agent, env, episode_length=6))
    # print(mdp.get_feature_expectations([trajectory]))


    # Set up inference
    # true_reward_given = np.array([0,0,1,0])
    reward_space_true = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]
    reward_space_proxy = reward_space_true
    len_reward_space = len(reward_space_true)
    # reward_space = [np.array([1,0]),np.array([0,1]), np.array([1,1])]
    inference = Inference(agent, env, beta=beta, reward_space_true=reward_space_true,
                          num_traject=10, prior=None)


    'Set up test environment (not used)'
    # print 'starting posterior calculation'
    # inference.calc_and_save_posterior(proxy_given, reward_space_proxy)
    # prior = dict([(tuple(true_reward), inference.get_posterior(true_reward))
    #               for true_reward in reward_space_true])
    # print('new prior: {prior}'.format(prior=prior))
    # mdp_test = NStateMdpGaussianFeatures(num_states=num_states, rewards=proxy_given, start_state=0, preterminal_states=[],   # proxy_given should have no effect
    #                                      feature_dim=feature_dim, num_states_reachable=num_states, SEED=SEED)
    # mdp_test.add_feature_map(mdp.features)
    # env_test = GridworldEnvironment(mdp_test)
    # agent.set_mdp(mdp_test)
    # inference_test = Inference(agent, env_test, beta=1., reward_space_true=reward_space_true, num_traject=1, prior=prior)



    'Calculate proposal set'
    # cost_of_asking = 0.01
    # best_r_set = []; best_r_set_gain = 0
    set_of_proposal_sets = powerset(reward_space_proxy)
    # for reward in set_of_proposal_sets:
    #     print(reward)
    # print(list(set_of_proposal_sets))
    # TODO: Mind the +1!
    uniform_prior = {tuple(reward): np.true_divide(1,len_reward_space+1) for reward in reward_space_true}
    uniform_prior[(1,0,0,0)] = 0.5
    best_r_set, best_r_set_regret, best_regret_plus_cost, best_posterior, best_post_avg = choose_regret_minimizing_proposal(set_of_proposal_sets,
                                                                    reward_space_true, prior=uniform_prior, inference=inference)

    'Print results'
    print 'mdp_features:'
    print np.array([np.concatenate([[state], mdp.features[state]]) for state in range(num_states)])
    # # print('mdp features: {features}'.format(features=mdp_test.features))
    # print(avg_weighted_reward)
    # print(exp_reward_after_asking)
    print(best_r_set_regret)
    print(best_r_set)
    print('Best post_avg:{post_avg}').format(post_avg=best_post_avg)
    print('Best posterior:{posterior}').format(posterior=best_posterior)



    'Create interface'
    # omega = best_r_set # replace with chosen omega
    # interface = Interface(omega, agent, env_test, num_states=num_states)
    # interface.plot()



    """Todo:
    # Two notes:
        1) Companies may be the first major transformative AI applications that require agent-like AI. As Drexler has argued, many functions of AI could be implemented as AI services.
        2) Oligopolistic tendencies in AI may be inherited from oligopolistic tendencies in tech sectors. Research startups and academic labs exist, so the barriers to entry for research appear low. Profitability may require more data and a large existing customer base if the profit per customer is small. But data-efficient machine learning seems likely to improve strongly prior to transformative AI due to advances in transfer learning, continual learning, model-based reinforcement learning and unsupervised learning.

    # TODO: Calculate joint of w, tilde(w)
        -P(w') = E_w P(w' | w)P(w)
        -Calculation of P(w' | w):
            -Cache feature expectations with every w' in Q.
            -For every w:
                -Get likelihood P(w' | w). Done.
            -Or just P(w') = sum_w prior(w) * get_likelihood(proxy, true)
    # TODO: Fixed size queries (ca 4)
        -Test
        -Increase feature_dim
        -Test: Compare binary comparisons
        -Scale to sets of 4
    # TODO: Compare greedy lookahead against random query. Test by sampling a real reward from the prior and seeing how we do on average.
    # TODO: line-by-line profiler: https://github.com/rkern/line_profiler or https://plugins.jetbrains.com/plugin/8525-python-profiler-experimental
        # Github has a python file to run my files
        # python
    # TODO: Third task: Make sure to only do planning for each proxy once. Maybe make this function get the feature expectations instead and cache it.
    -Implement regret minimizer
        -Test inference first
        -Why stochasticity in choice and regret?
        -Test: Does the omega choice make sense?
    -Non-static features
    -Posterior not adding up - fix?
       -Test if new inference adds up
    -Interface:
        (-Visualize feature dist)
    -Try not sampling actions - Change back!
    -Other approaches/features for tomorrow
            + outline code (and section?)
    -Implement Race car / Dorsa domain with between-track generalization
    """


    """
    Proposal choosing:
    Exploitation = rewards with high posterior
    Exploration  = some sort of info gain in the set? Minimizing posterior-avg-reward?
    """











    """
    Evaluation depends on tie sampling! That could cause the posterior to not add up to 1!

    The agent can choose the preterminal state because it gives equal immediate reward, but then it'll
    get less average reward because it has spent more relative time in the shitty start state.
    Since the time where the terminal state is entered is random, so is the avg reward and the likelihood.

    But if we choose best_actions[0], is there a problem? Apparently not (anymore?).
    Also, on average the inference gives 1.0.
    """
