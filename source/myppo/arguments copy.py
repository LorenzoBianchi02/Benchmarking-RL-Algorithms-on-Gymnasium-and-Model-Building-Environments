import argparse

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')
	parser.add_argument('--total_timesteps', dest='total_timesteps', type=int, default=10000000)
	parser.add_argument('--gym_env', dest='gym_env', type=str, default='Pendulum-v1')
	parser.add_argument('--render_mode', dest='render_mode', type=str)

	args = parser.parse_args()

	return args
