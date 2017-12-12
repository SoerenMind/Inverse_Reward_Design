import numpy as np
from agent_runner import run_agent

class InferenceOld:
    def __init__(self,agent,beta,reward_space):
        # Or input env_type+s_terminal+etc
        # self.env = env
        self.agent = agent
        self.beta = beta
        self.reward_space = reward_space
    def get_prior(self, rfunc_true):
        return np.true_divide(1, len(self.reward_space))
    def get_likelihood(self, rfunc_true, rfunc_proxy):
        self.agent.add_rfunc(rfunc_proxy)
        expected_avg_reward = self.agent.get_avg_true_reward(rfunc_true)
        return np.exp(self.beta*expected_avg_reward)
    def get_Z_constant(self,rfunc_true):
        Z_normalization = 0
        for rfunc_proxy in self.reward_space:
            Z_normalization += self.get_likelihood(rfunc_true, np.array(rfunc_proxy)) \
                               * self.get_prior(rfunc_proxy)
        return Z_normalization
    def get_posterior(self, rfunc_true, rfunc_proxy):
        '''Just Bayes' rule'''
        lhood = self.get_likelihood(rfunc_true,rfunc_proxy)
        Z = self.get_Z_constant(rfunc_true)
        prior = self.get_prior(rfunc_true)
        if Z == 0: print('Warning: Z=0')
        return np.true_divide(lhood,Z) * prior


class Inference:
    def __init__(self,agent,beta,reward_space):
        # Or input env_type+s_terminal+etc
        # self.env = env
        self.agent = agent
        self.beta = beta
        self.reward_space = reward_space
    def get_prior(self, rfunc_true):
        return np.true_divide(1, len(self.reward_space))
    def get_likelihood(self, rfunc_true, rfunc_proxy):
        """Calculates likelihodd of proxy reward given true reward. Agent optimizes for proxy reward and reports
        expected true reward.
        Options:
        -Generate trajectories with agent runner+proxy_reward argument.
        """
        self.agent.add_rfunc(rfunc_proxy)
        expected_avg_reward = self.agent.get_avg_true_reward(rfunc_true)
        return np.exp(self.beta*expected_avg_reward)
    def get_Z_constant(self,rfunc_true):
        Z_normalization = 0
        for rfunc_proxy in self.reward_space:
            Z_normalization += self.get_likelihood(rfunc_true, np.array(rfunc_proxy)) \
                               * self.get_prior(rfunc_proxy)
        return Z_normalization
    def get_posterior(self, rfunc_true, rfunc_proxy):
        '''Just Bayes' rule'''
        lhood = self.get_likelihood(rfunc_true,rfunc_proxy)
        Z = self.get_Z_constant(rfunc_true)
        prior = self.get_prior(rfunc_true)
        if Z == 0: print('Warning: Z=0')
        return np.true_divide(lhood,Z) * prior


class Determ_Inference(InferenceOld):
    '''for testing purposes'''
    def get_likelihood(self, rfunc_true, rfunc_proxy):
        self.agent.add_rfunc(rfunc_proxy)
        expected_avg_reward = self.agent.get_avg_true_reward(rfunc_true)
        return expected_avg_reward


def test_inference(inference, rfunc_proxy_given, reward_space):
    '''Tests if posterior adds up to 1 by calculating it for every possible true reward function.'''
    cum_post = 0
    # for rfunc_true in itertools.product([0,1],repeat=num_states):
    for rfunc_true in reward_space:
        post = inference.get_posterior(rfunc_true, rfunc_proxy_given)
        cum_post += post
    return cum_post



if __name__=='__main__':
    rewards = [0, 1, 2, 3, 4]
    mdp = NStateMdp(num_states=5, rewards=rewards, start_state=0, preterminal_states=[3])
    env = GridworldEnvironment(mdp)
    # default_action = 1
    # agent = DirectionalAgent(default_action)
    agent = ImmediateRewardAgent()
    agent.set_mdp(mdp)
    print(run_agent(agent, env, episode_length=float(6)))