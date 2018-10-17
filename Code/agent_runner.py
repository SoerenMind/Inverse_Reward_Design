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
