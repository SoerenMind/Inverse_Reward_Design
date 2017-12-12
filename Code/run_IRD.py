import numpy as np
import environment
import agent_class
from inference_class import InferenceOld, Determ_Inference, test_inference
import itertools
from gridworld import NStateMdp, GridworldEnvironment, Direction
from agents import ImmediateRewardAgent, DirectionalAgent
from agent_runner import run_agent






if __name__=='__main__':
    # Define environment and agent
    rewards = [0,0,1]
    mdp = NStateMdp(num_states=3, rewards=rewards, start_state=0, preterminal_states=[3])
    env = GridworldEnvironment(mdp)
    agent = ImmediateRewardAgent()
    agent.set_mdp(mdp)
    print(run_agent(agent, env, episode_length=float(6)))

    # Set up inference
    rfunc_proxy_given = rewards
    rfunc_true_given = np.array([0,0,1])
    reward_space = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
    # env = environment.N_State_Env(s_terminal=[2], horizon=1000, s_start=0)
    # agent = agent_class.One_Step_Planner(env)
    # agent.add_rfunc(rfunc_proxy_given)
    inference = Determ_Inference(agent, beta=1., reward_space=reward_space)












    # def test_grid_world():
#     '''unfinished'''
#     width = 4; height = 2
#     target = np.array([3,1]); s_start = np.array([0,0])
#     rfunc_proxy = np.zeros([width,height]) - 0.1
#     rfunc_proxy[target[0]][target[1]] = 1
#     env = environment.Basic_Grid_World(height, width, s_terminal=[target], horizon = 10, s_start=s_start)
#     agent = agent_class.Basic_Grid_Walker_Up_Right(env)
#     agent.add_rfunc(rfunc_proxy)
#     agent.get_next_action(s_start)
#     return agent.get_trajectory()
#     # TODO(later): add do_action to environment class and try with n-step MDP

# TODO: Implement Inference meta class
# TODO: Randomize trajectories
# if __name__ == '__main__':
#     rfunc_proxy_given = np.array([0, 0, 1])
#     rfunc_true_given = np.array([0,0,1])
#     reward_space = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
#     env = environment.N_State_Env(s_terminal=[2], horizon=1000, s_start=0)
#     agent = agent_class.One_Step_Planner(env)
#     agent.add_rfunc(rfunc_proxy_given)
#     inference = inference_class.Determ_Inference(agent, beta=1., reward_space=reward_space)
#
#     # print(test_grid_world())
#
#
#     # print(agent.get_trajectory())
#     # print(agent.get_feature_expectations())
#     # print(agent.get_avg_true_reward(rfunc_true_given))
#     # print(inference.get_likelihood(rfunc_true_given, rfunc_proxy_given))
#     # print(inference.get_Z_constant(rfunc_true_given))
#     # print(inference.get_prior(rfunc_true_given))
#     # print(inference.get_posterior(rfunc_true_given, rfunc_proxy_given))
#     print(inference_class.test_inference(inference, rfunc_proxy_given, reward_space))