from plotter import plot
from train import train
import os

algs = ['PPO', 'PPO']
env = 'LunarLanderContinuous-v2'

i = 9
for alg in algs:
    train(env, alg)
    original_path = f"{alg}/{env}/progress.csv"
    new_path = f"{alg}/{env}/progress{i}.csv"
    os.rename(original_path, new_path)
    i+=1