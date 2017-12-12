import numpy as np

class Agent:
    def __init__(self,env):
        self.env = env
        # self.state = self.env.s_start
    def add_rfunc(self,rfunc):
        '''Override method should learn a policy for rfunc'''
        self.env.add_rfunc(rfunc)
    def get_trajectory(self):
        pass
    def get_next_action(self,s):
        pass
    def get_feature_expectations(self):
        pass
    def get_avg_true_reward(self,rfunc_true):
        '''Given the rfunc_proxy in the environment, returns true avg reward'''
        # TODO: Save avg_true_reward | proxy for every rfunc_true or save planned policy to calculate Z
        return np.dot(self.get_feature_expectations(), rfunc_true)

class One_Step_Planner(Agent):
    '''Takes action that will maximize reward at next state.
    Input:
        -reward function (rfunc)
        -environment
    Methods:
        -get_next_action
        -get_trajectory
        -get_feature expectations
    '''
    def get_next_action(self,s):
        immediate_rewards = self.env.get_rfunc()
        action = np.argmax(immediate_rewards)   # Add breaking ties
        assert isinstance(action,int)
        return action
    def get_trajectory(self):
        '''Generates trajectory and returns state counts and steps taken, then resets steps to 0.'''
        # TODO: env.copy OR expectations inside trajectory OR return steps taken
        assert self.env.steps_taken == 0
        s = self.env.s_start   # could do self.state
        state_counts = np.zeros(self.env.get_num_states())
        while not self.env.is_terminal(s):
            state_counts[s] += 1
            a = self.get_next_action(s)
            s, r = self.env.get_next_state_reward_update(s,a)
        state_counts[s] += 1
        steps_taken = self.env.steps_taken
        self.env.steps_taken = 0
        return state_counts, steps_taken
    def get_feature_expectations(self):
        '''Calls self.get_trajectory; returns feature expectations.'''
        state_counts, steps_taken = self.get_trajectory()
        feature_expectations = np.true_divide(state_counts, steps_taken + 1)
        assert np.sum(feature_expectations) == 1
        return feature_expectations

class One_Step_Planner_No_Start_State(One_Step_Planner):
    def get_trajectory(self):
        '''Generates trajectory and returns state counts and steps taken, then resets steps to 0.
        Doesn't record start state in trajectory to make sure avg reward = 1 is possible.'''
        # TODO: env.copy OR expectations inside trajectory OR return steps taken
        assert self.env.steps_taken == 0
        s = self.env.s_start   # could do self.state
        state_counts = np.zeros(self.env.get_num_states())
        while not self.env.is_terminal(s):
            a = self.get_next_action(s)
            s, r = self.env.get_next_state_reward_update(s,a)
            state_counts[s] += 1
        steps_taken = self.env.steps_taken-1    # note change
        self.env.steps_taken = 0
        return state_counts, steps_taken

class Basic_Grid_Walker_Up_Right(Agent):
    def learn_policy(self):
        pass
    def get_next_action(self,state):
        if state[0] < self.env.width-1:
            return 1
        elif state[1] < self.env.height-1:
            return 2
        else:
            raise ValueError('attempting to move from terminal state')
        # TODO: Change all uses of s to state
    def get_trajectory(self):
        '''Generates trajectory and returns state counts and steps taken, then resets steps to 0.'''
        # TODO: env.copy OR expectations inside trajectory OR return steps taken
        assert self.env.steps_taken == 0
        s = self.env.s_start   # could do self.state
        (x,y) = s
        state_counts = np.zeros([self.env.width,self.env.height])
        while not self.env.is_terminal(s):
            state_counts[x][y] += 1
            a = self.get_next_action(s)
            s, r = self.env.get_next_state_reward_update(s,a)
            (x, y) = s
        state_counts[x][y] += 1
        steps_taken = self.env.steps_taken # copy?
        self.env.steps_taken = 0
        return state_counts, steps_taken
