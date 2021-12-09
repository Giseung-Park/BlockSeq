import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import argparse


def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w


def xy2updatenumy(x, y, update_timestep, max_timestep):
	arr_x = []
	arr_y = []
	for i in range(int(np.floor(max_timestep / update_timestep))):
		ind = np.where(x < update_timestep * (i + 1))[0]
		if ind.size > 0:
			arr_x.append(update_timestep * (i + 1))
			arr_y.append(y[ind[len(ind) - 1]])
	return np.array(arr_x), np.array(arr_y)


def mat_plot(env_name='Pendulum-v0', steps=50000, trans_seq=32, seed_num=5, R_val=15.0):
	arr_x = []
	arr_y = []
	minlen = 1e10
	minind = -1
	for seed in range(seed_num):
		
		if (env_name == 'MountainHike') or (env_name == 'Pendulum-v0'):
			update_steps = 200
			mat = scipy.io.loadmat('data/' + env_name + '/' + env_name + '_L=' + str(trans_seq) + '_seed_' + str(seed)
			                       + '_steps_' + str(steps) + '_proposed.mat')
		elif env_name == 'Sequential_Rchange':
			update_steps = 128
			mat = scipy.io.loadmat('data/' + env_name + '/R='+ str(R_val) + '/' + env_name + '_L=' + str(trans_seq) + '_seed_' + str(seed)
			                       + '_steps_' + str(steps) + '_proposed.mat')
		else:
			raise NotImplementedError
		
		
		mat = {k: v for k, v in mat.items() if k[0] != '_'}
		
		## Step 1. Find nonzero step episodes(cumulative) and episode reward. Delete dummy.
		x_axis_steps = mat['steps'].squeeze()
		nonzero_inds = np.where(x_axis_steps > 0)[0]
		
		x_axis_steps = np.cumsum(x_axis_steps[nonzero_inds])
		y_axis_epi_rew = mat['R'].squeeze()[nonzero_inds]
		
		## Step 2. WINDOW
		x_axis_steps = x_axis_steps[99:]
		y_axis_epi_rew = moving_average(y_axis_epi_rew, 100)
		
		## Step 3. Synchronize x_axis for every seed.
		x_axis_steps, y_axis_epi_rew = xy2updatenumy(x_axis_steps, y_axis_epi_rew, update_timestep=update_steps, max_timestep=steps)
		
		if minlen > len(x_axis_steps):
			minlen = np.min([minlen, len(x_axis_steps)])
			minind = seed
		
		arr_x.append(x_axis_steps)
		arr_y.append(y_axis_epi_rew)
	
	minlen = int(minlen)
	x_final = arr_x[minind]
	for seed in range(seed_num):
		len_t = len(arr_y[seed])
		arr_y[seed] = arr_y[seed][len_t - minlen:len_t]

	meany = np.mean(arr_y, axis=0)
	stdy = np.std(arr_y, axis=0)
	
	plot, = plt.plot(x_final, meany)
	yl = meany - stdy
	yu = meany + stdy
	plt.fill_between(x_final, yl, yu, alpha=0.2)
	
	return plot



parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--steps', type=int, default=50000)
parser.add_argument('--trans_seq_interval', type=int, default=32)
parser.add_argument('--seed_num', type=int, default=5)
# seed num=n: 0, 1, ..., n-1.
parser.add_argument('--R_value', type=float, default=15.0)


args = parser.parse_args()
env_name = args.env
max_steps = args.steps
R_value = args.R_value

arr_plot = []
arr_plot.append( mat_plot(env_name=env_name, steps=max_steps, trans_seq=args.trans_seq_interval, seed_num=args.seed_num, R_val=R_value) )


if env_name == 'MountainHike':
	plt.title('MountainHike', fontsize=20)
elif env_name == 'Pendulum-v0':
	plt.title('Pendulum, random missing', fontsize=20)
elif env_name == 'Sequential_Rchange':
	plt.title('Sequential, R=' + str(int(args.R_value)), fontsize=20)
else:
	raise NotImplementedError


plt.xlim(right=max_steps)

plt.xlabel("Timestep", fontsize=20)
plt.ylabel("Average Return", fontsize=20)
plt.xticks(size=14)
plt.yticks(size=14)
plt.tight_layout()

# plt.show()
plt.savefig('data/' + env_name + '.png')




