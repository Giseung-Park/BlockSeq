import numpy as np
import torch
import os, argparse, warnings
import scipy.io as sio

from learning_models import Block_model, RL_agent
from attention_model import BaseModel

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--steps', type=int, default=50000)
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--trans_seq_interval', type=int, default=32)
parser.add_argument('--pre_train_times', type=int, default=500)
parser.add_argument('--rws_sample_number_K', type=int, default=50)
parser.add_argument('--trans_num_layer', type=int, default=2)
parser.add_argument('--norm_type', type=str, default='post')
parser.add_argument('--train_step_st', type=int, default=5)
parser.add_argument('--train_step_rl', type=int, default=1)
parser.add_argument('--top_k', type=int, default=2)
parser.add_argument('--R_value', type=float, default=15.0)

args = parser.parse_args()
env_name = args.env

if (env_name == 'MountainHike') or (env_name == 'Pendulum-v0'):
	savepath = './data/' + args.env  + '/'
elif env_name == 'Sequential_Rchange':
	savepath = './data/' + args.env + '/R='+ str(args.R_value) + '/'
else:
	raise NotImplementedError



if os.path.exists(savepath):
	warnings.warn('{} exists (possibly so do data).'.format(savepath))
else:
	os.makedirs(savepath)


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

beta_h = 'auto_1.0'
optimizer_st = 'adam'
minibatch_size = 4
seq_len = 64
reward_scale = 1.0
lr_vrm = 8e-4
gamma = 0.99
max_all_steps = args.steps

if args.render:
	rendering = True
else:
	rendering = False
	
	
## Environments
action_filter = lambda a: a.reshape([-1])

if env_name == "MountainHike":
	from environments.mountain_hike.death_valley import DeathValleyEnv
	
	env = DeathValleyEnv(observation_var=0.03, max_action_value=0.01, max_time=200)
	
	max_steps = 200
	est_min_steps = 200
	
elif env_name == "Pendulum-v0":
	from environments.env_wrapper.pomdp_wrapper import POMDPWrapper
	
	env = POMDPWrapper(env_name)
	
	max_steps = 200
	est_min_steps = 199
	
elif env_name == "Sequential_Rchange":
	from environments.ori_task import TaskT
	
	env = TaskT(3, R=args.R_value)
	
	max_steps = 128
	est_min_steps = 10

else:
	raise NotImplementedError
	
	

rnn_type = 'mtgru'
d_layers = [256, ]
z_layers = [64, ]
x_phi_layers = [128]

value_layers = [256, 256]
policy_layers = [256, 256]

step_start_rl = 1000
step_start_st = 1000
step_end_st = np.inf
pre_train_times = args.pre_train_times

train_step_rl = args.train_step_rl
train_step_st = args.train_step_st

train_freq_rl = 1. / train_step_rl
train_freq_st = 1. / train_step_st

max_episodes = int(max_all_steps / est_min_steps) + 1  # for replay buffer


### Model declaration
attention_hidden_dim = 256
trans_seq_interval = args.trans_seq_interval

trans_model_q = BaseModel(hidden_dim=attention_hidden_dim, num_layers=args.trans_num_layer,
                          norm_type=args.norm_type)

block_model = Block_model(trans_model_q, input_size=env.observation_space.shape[0] + 1,
          action_size=env.action_space.shape[0],
          rnn_type=rnn_type,
          d_layers=d_layers,
          z_layers=z_layers,
          x_phi_layers=x_phi_layers,
          optimizer=optimizer_st,
          lr_st=lr_vrm, sample_num_K=args.rws_sample_number_K, top_k=args.top_k)

agent = RL_agent(block_model, gamma=gamma,
             beta_h=beta_h,
             value_layers=value_layers,
             policy_layers=policy_layers)


SP_real = np.zeros([max_episodes, max_steps, env.observation_space.shape[0]], dtype=np.float32)  # observation (t+1)
A_real = np.zeros([max_episodes, max_steps, env.action_space.shape[0]], dtype=np.float32)  # action
R_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # reward
D_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # done
V_real = np.zeros([max_episodes, max_steps],
                  dtype=np.float32)  # mask, indicating whether a step is valid. value: 1 (compute gradient at this step) or 0 (stop gradient at this step)

e_real = 0
global_step = 0
loss_sts = []

#  Run
agent.train()
while global_step < max_all_steps:
	
	print("global step:", global_step)
	
	s0 = env.reset().astype(np.float32)
	r0 = np.array([0.], dtype=np.float32)
	x0 = np.concatenate([s0, r0])
	a = agent.init_episode(x0).reshape(-1)
	
	for t in range(max_steps):
		
		if global_step == max_all_steps:
			break
		
		if np.any(np.isnan(a)):
			raise ValueError
		
		sp, r, done, _ = env.step(action_filter(a))
		if rendering:
			env.render()
		
		A_real[e_real, t, :] = a
		SP_real[e_real, t, :] = sp.reshape([-1])
		R_real[e_real, t] = r
		D_real[e_real, t] = 1 if done else 0
		V_real[e_real, t] = 1
		
		### We assume that trans_seq_interval is not 1.
		agent.eval()
		
		if t % trans_seq_interval < trans_seq_interval - 1:
			a = agent.select(sp, r)
			
		else: #  t % trans_seq_interval == trans_seq_interval - 1. block length is trans_seq_interval*N.
			a = agent.select(sp, r, SP_real[e_real, t - t % trans_seq_interval: t + 1, :],
			                 R_real[e_real, t - t % trans_seq_interval: t + 1],
			                 A_real[e_real, t - t % trans_seq_interval: t + 1, :])
		agent.train()
		
		global_step += 1
		
		if global_step == step_start_st + 1 and pre_train_times > 0:
			print("Start pre-training the model!")
			_, _, loss_st = agent.learn_st(SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
			                               D_real[0:e_real], V_real[0:e_real],
			                               times=pre_train_times, minibatch_size=minibatch_size,
			                               trans_seq_interval=trans_seq_interval)
			loss_sts.append(loss_st)
			print("Finish pre-training the model!")
		
		if global_step > step_start_st and global_step <= step_end_st and np.random.rand() < train_freq_st:
			_, _, loss_st = agent.learn_st(SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
			                               D_real[0:e_real], V_real[0:e_real],
			                               times=max(1, int(train_freq_st)), minibatch_size=minibatch_size,
			                               trans_seq_interval=trans_seq_interval)
			loss_sts.append(loss_st)
		
		if global_step > step_start_rl and np.random.rand() < train_freq_rl:
			if global_step == step_start_rl + 1:
				print("Start training the RL controller!")
			agent.learn_rl_sac(SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
			                   D_real[0:e_real], V_real[0:e_real],
			                   times=max(1, int(train_freq_rl)), minibatch_size=minibatch_size,
			                   reward_scale=reward_scale, seq_len=seq_len, trans_seq_interval=trans_seq_interval)

		if done:
			break

	print(env_name + " -- episode {} : steps {}".format(e_real, t+1 ))
	e_real += 1

loss_sts = np.reshape(loss_sts, [-1, 2]).astype(np.float64)

data = {"loss_sts": loss_sts,
        "max_episodes": max_episodes,
        "step_start_rl": step_start_rl,
        "step_start_st": step_start_st,
        "step_end_st": step_end_st,
        "rnn_type": rnn_type,
        "optimizer": optimizer_st,
        "reward_scale": reward_scale,
        "beta_h": beta_h,
        "minibatch_size": minibatch_size,
        "train_step_rl": train_step_rl,
        "train_step_st": train_step_st,
        "R": np.sum(R_real, axis=-1).astype(np.float64),
        "steps": np.sum(V_real, axis=-1).astype(np.float64)}

sio.savemat(savepath + env_name + "_L=" + str(trans_seq_interval) + "_seed_" + str(seed) + "_steps_" + str(args.steps) + "_proposed.mat", data)
torch.save(agent, savepath + env_name + "_L=" + str(trans_seq_interval) + "_seed_" + str(seed) + "_steps_" + str(args.steps) + "_proposed.model")

