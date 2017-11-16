import numpy as np


class Environment:
    def __init__(self,rfunc,s_terminal=np.array([]),horizon=1,s_start=0):
        self.rfunc = rfunc
        self.s_terminal = s_terminal    # terminal states array
        self.horizon = horizon
        self.steps_taken = 0
        self.s_start = s_start
    def trans(self,s,a):
        # self.steps_taken += 1
        pass
    def reward(self,s,a):
        pass
    def next_state_reward(self,s,a):
        return self.trans(s,a), self.reward(s,a)
    def next_state_reward_update(self,s,a):
        self.steps_taken += 1
        return self.next_state_reward(s,a)
    def is_terminal(self,s):
        return s in self.s_terminal or self.steps_taken >= self.horizon - 1
    def num_states(self):
        return self.rfunc.shape[0]

class Two_State_Env(Environment):
    def trans(self,s,a):
        if not isinstance(a,int):
            print(Warning('a should be int'))
        return a
    def reward(self,s,a):
        return self.rfunc[a]
    def avg_reward(self,feature_expectations):
        return np.dot(feature_expectations,rfunc)


class Agent:
    def __init__(self,rfunc,env):
        self.rfunc = rfunc
        self.env = env
        # self.state = self.env.s_start
    def trajectory(self):
        pass
    def next_action(self,s):
        pass
    def feature_expectations(self):
        pass

class One_Step_Planner(Agent):
    '''Takes action that will maximize reward at next state.
    Input:
        -reward function (rfunc)
        -environment
    Methods:
        -next_action
        -trajectory
        -feature expectations
    '''
    def next_action(self,s):
        immediate_rewards = self.rfunc
        action = np.argmax(immediate_rewards)   # Add breaking ties
        assert isinstance(action,int)
        return action
    def trajectory(self):
        '''Generates trajectory and returns state counts and steps taken, then resets steps to 0.'''
        # TODO: env.copy OR expectations inside trajectory OR return steps taken
        s = self.env.s_start   # could do self.state
        state_counts = np.zeros(self.env.num_states())
        while not self.env.is_terminal(s):
            state_counts[s] += 1
            a = self.next_action(s)
            s, r = self.env.next_state_reward_update(s,a)
        state_counts[s] += 1
        steps_taken = self.env.steps_taken
        self.env.steps_taken = 0
        return state_counts, steps_taken
    def feature_expectations(self):
        '''Calls self.trajectory; returns feature expectations.'''
        state_counts, steps_taken = self.trajectory()
        return np.true_divide(state_counts, steps_taken + 1)
    def avg_agent_reward(self):
        return self.env.avg_reward(self.feature_expectations())


if __name__ == '__main__':
    rfunc = np.array([0,0])
    env = Two_State_Env(rfunc,s_terminal=np.array([]),horizon=4,s_start=0)
    agent = One_Step_Planner(rfunc,env)
    print(agent.trajectory())
    print(agent.feature_expectations())
