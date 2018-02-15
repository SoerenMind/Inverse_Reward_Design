from agents import DirectionalAgent, ImmediateRewardAgent, ValueIterationLikeAgent
from gridworld import GridworldMdp , GridworldEnvironment, Direction, NStateMdp, GridworldMdpWithDistanceFeatures
import numpy as np


def run_agent(agent, env, episode_length=20):
    """Runs the agent on the environment for one episode.

    The agent will keep being asked for actions until the environment says the
    episode is over, or once the episode_length has been reached.

    agent: An Agent (which in particular has get_action and inform_minibatch).
    env: An Environment in which the agent will act.
    episode_length: The maximum number of actions that the agent can take. If
    the agent has not reached a terminal state by this point, the episode is
    terminated early.

    Returns the trajectory that the agent took, which is a list of (s, a, s', r)
    tuples.
    """
    env.reset()
    trajectory = []
    # Todo: Trajectory will include terminal state iff episode length not reached. Then the final next_state won't be rewarded.
    # Tweak: Exclude start state from trajectory?
    # agent.compute_policy()
    while len(trajectory) < episode_length and not env.is_done():
        curr_state = env.get_current_state()
        action = agent.get_action(curr_state)
        # action = agent.quick_get_action(curr_state)
        next_state, reward = env.perform_action(action)
        minibatch = (curr_state, action, next_state, reward)
        agent.inform_minibatch(*minibatch)
        trajectory.append(minibatch)
    return trajectory


if __name__=='__main__':
    # Parameters
    dist_scale = 1
    living_reward = -0.01
    noise = 0
    width = 8; height = 8

    # Create environment
    # goals = [(1,1), (3,3), (3,4), (4,5), (6,4), (6,6)]
    goals = [(1,1), (2,6), (3,3), (3,4), (4,5), (6,4), (6,6)]
    rewards = np.zeros(6)
    grid = GridworldMdp.generate_random(8,8,0.1,0.1,goals,living_reward=-0.01)
    # mdp = GridworldMdp(grid,-0.01,0)
    mdp = GridworldMdpWithDistanceFeatures(grid, dist_scale, living_reward, noise, rewards)
    env = GridworldEnvironment(mdp)

    agent = ValueIterationLikeAgent(gamma=0.8, num_iters=50)
    agent.set_mdp(mdp)  # Does value iteration
    vals = np.zeros([height,width])
    for x in range(1,width-1):
        for y in range(1,height-1):
            vals[y][x] = agent.values[x,y]
    print vals.round(1)

    # vals2 = np.empty([height,width])
    # for x in range(1,width-1):
    #     for y in range(1,height-1):
    #         vals[y][x] = agent.values[y,x]
    # print (vals.round(2) / vals.__abs__().max().max()).round(1)


    print run_agent(agent, env, episode_length=20)
    # rewards = [0, 1, 2, 3, 4]
    # mdp = NStateMdp(num_states=5, rewards=rewards, start_state=0, preterminal_states=[3])
    # env = GridworldEnvironment(mdp)
    # default_action = 1
    # # agent = DirectionalAgent(default_action)
    # agent = ImmediateRewardAgent()
    # agent.set_mdp(mdp)
    # print(run_agent(agent, env, episode_length=6))
