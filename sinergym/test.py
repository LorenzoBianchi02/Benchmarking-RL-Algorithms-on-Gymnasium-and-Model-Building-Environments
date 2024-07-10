import gymnasium as gym
import numpy as np
import sinergym
from sinergym.utils.wrappers import (LoggerWrapper, MultiObsWrapper,
                                     NormalizeObservation)


from sinergym.utils.logger import CSVLogger
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

class CustomCSVLogger(CSVLogger):

    def __init__(
            self,
            monitor_header: str,
            progress_header: str,
            log_progress_file: str,
            log_file: Optional[str] = None,
            flag: bool = True):
        super(CustomCSVLogger, self).__init__(monitor_header,progress_header,log_progress_file,log_file,flag)
        self.last_10_steps_reward = [0]*10

    def _create_row_content(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            terminated: bool,
            truncated: bool,
            info: Optional[Dict[str, Any]]) -> List:
            
        if info.get('reward') is not None:
            self.last_10_steps_reward.pop(0)
            self.last_10_steps_reward.append(info['reward'])

        
        return [
            info.get('timestep',0)] + list(obs) + list(action) + [
            info.get('time_elapsed(hours)',0),
            info.get('reward',None),
            np.mean(self.last_10_steps_reward),
            info.get('total_power_no_units'),
            info.get('comfort_penalty'),
            info.get('abs_comfort'),
            terminated,
            truncated]

env=gym.make('Eplus-demo-v1')
env=LoggerWrapper(env,logger_class=CustomCSVLogger,monitor_header = ['timestep'] + env.get_wrapper_attr('observation_variables') +
                env.get_wrapper_attr('action_variables') + ['time (hours)', 'reward', '10-mean-reward',
                'power_penalty', 'comfort_penalty', 'terminated', 'truncated'])

for i in range(1):
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    current_month = 0
    while not (terminated or truncated):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print('Episode ', i, 'Mean reward: ', np.mean(
        rewards), 'Cumulative reward: ', sum(rewards))
env.close()