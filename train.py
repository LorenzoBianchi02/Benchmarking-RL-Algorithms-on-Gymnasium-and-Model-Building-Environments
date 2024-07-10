import gymnasium as gym
import sinergym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3 import SAC
from stable_baselines3 import A2C
from stable_baselines3 import DDPG


from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from sinergym.utils.wrappers import (LoggerWrapper, NormalizeAction,
                                     NormalizeObservation)


def train(env, alg, eval_freque=400):
    print(alg + " " + env)
    print("starting...")


    vec_name = env
    model_name = alg
    save_loc = model_name + "/" + vec_name
    save_loc_buf = save_loc + "/rep_buff"


    ###### LOGGER ######
    new_logger = configure(save_loc, ["stdout", "csv"])


    #print("making eval callback...")
    ####### EVAL #######
    #eval_env = Monitor(gym.make(vec_name))
    ##stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=30, verbose=0)
    #eval_callback = EvalCallback(eval_env, best_model_save_path=save_loc,
    #                            log_path=save_loc, eval_freq=eval_freque,
    #                            deterministic=True, render=False,
    #                            verbose=1)


    print("creating env...")
    #### ENV #####
    vec_env = gym.make(vec_name)
    #vec_env = NormalizeAction(vec_env)
    #vec_env = NormalizeObservation(vec_env)
    #vec_env = LoggerWrapper(vec_env)
    #vec_env = gym.make(vec_name, render_mode='human')

    ##### MODEL #######
    if alg == "PPO":
        #model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=1024, batch_size=64, gae_lambda=0.98, learning_rate=0.999, n_epochs=4, ent_coef=0.01)
        model = PPO("MlpPolicy", vec_env, verbose=0)
    elif alg == "SAC":
        #model = SAC("MlpPolicy", vec_env, verbose=0)
        model = SAC("MlpPolicy", vec_env, verbose=0, learning_rate=3e-4, buffer_size=1000000, batch_size=256, ent_coef=0.1, train_freq=1, gradient_steps=1, gamma=0.99, tau=0.01, learning_starts=10000)
    elif alg == "TD3":
        model = TD3("MlpPolicy", vec_env, verbose=0, gamma=0.98, buffer_size=200000, learning_starts=10000, gradient_steps=1, train_freq=1, learning_rate=1e-3)
    elif alg == "DQN":
        model = DQN("MlpPolicy", vec_env, verbose=0, learning_rate=6.3e-4, batch_size=128, buffer_size=50000, learning_starts=0, gamma=0.99, target_update_interval=250, train_freq=4, gradient_steps=-1, exploration_fraction=0.12, exploration_final_eps=0.1)
    elif alg == "A2C":
        model = A2C("MlpPolicy", vec_env, verbose=0)
    elif alg == "DDPG":
        model = DDPG("MlpPolicy", vec_env, verbose=0)
    else:
        print("ALG NOT FOUND")
        return
    #model = TD3.load(save_loc + "/best_model", env=vec_env)

    model.set_logger(new_logger)
    #model.load_replay_buffer(save_loc_buf)


    print("training...")

    model.learn(total_timesteps=10000000, reset_num_timesteps=False)
    #model.save(save_loc)
    #model.save_replay_buffer(save_loc_buf)
    #params = model.get_parameters()

    vec_env.close()
    #exit()

    return
