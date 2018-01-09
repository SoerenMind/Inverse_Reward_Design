import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from agent_runner import run_agent



def pad(seq, target_length, padding=0):
    """Extend the sequence seq with padding (default: 0) so as to make
    its length up to target_length. Return seq. If seq is already
    longer than target_length, raise ValueError."""
    length = len(seq)
    if length > target_length:
        raise ValueError("sequence too long ({}) for target length {}"
                           .format(length, target_length))
    seq.extend([padding] * (target_length - length))
    return seq



class Interface(object):
    def __init__(self, omega, agent, env, num_states, num_traject=1):
        self.omega = omega
        self.agent = agent
        self.env = env
        self.num_states = num_states
        self.t = np.arange(0,num_states,1)
        self.num_traject = num_traject

    def get_state_freq(self, selection):
        """Returns the selected reward"""
        reward = self.omega[int(selection)]
        self.agent.mdp.change_reward(reward)
        trajectories = [run_agent(self.agent, self.env) for _ in range(self.num_traject)]
        state_list = self.agent.mdp.get_state_list(trajectories)
        # if selection <= 0.5: state_list = [0,1,1,2]
        # if selection >= 0.5: state_list = [0,1,2,2]
        state_frequencies = np.bincount(state_list)
        state_frequencies = pad(list(state_frequencies), self.num_states)
        return state_frequencies

    def get_feature_dist(self, selection):
        pass

    def plot(self):
        axis_color = 'lightgoldenrodyellow'

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Adjust the subplots region to leave some space for the sliders and buttons
        fig.subplots_adjust(left=0.25, bottom=0.25)

        # Draw initial plot
        selection_0 = 0
        [self.line] = ax.plot(self.t, self.get_state_freq(selection_0))

        reward_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axis_color)
        self.reward_slider = Slider(reward_slider_ax, 'Reward', -0.2, len(self.omega)-0.8,
                                    valinit=selection_0, dragging=True)

        # Define an action for modifying the line when any slider's value changes
        self.reward_slider.on_changed(self.slider_on_changed)

        plt.show()

    def slider_on_changed(self,selection):
        self.line.set_ydata(self.get_state_freq(selection))
        # fig.canvas.draw_idle()


if __name__=='__main__':
    interface = Interface(omega=np.array([0,1]), agent=None, env=None, num_states=3)
    interface.plot()
