from train import train
import os

algs = ['TD3', 'PPO']
env = 'Eplus-5zone-hot-continuous-stochastic-v1'

i = 9
for alg in algs:
    train(env, alg)
    original_path = f"{alg}/{env}/progress.csv"
    new_path = f"{alg}/{env}/progress{i}.csv"
    os.rename(original_path, new_path)
    i+=1