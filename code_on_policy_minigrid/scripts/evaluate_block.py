import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
					help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
					help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
					help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
					help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
					help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
					help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
					help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=True,
					help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
					help="add a GRU to the model")

## Arguments for block generation
parser.add_argument("--block-length", type=int, default=8, help="Length L of each block B_n (L > 1)")
parser.add_argument("--top-k", type=int, default=4, help="Number of k selected elements in B_n. 1 <= k <= L.")
parser.add_argument('--block-hid-dim', type=int, default=256)
parser.add_argument('--att-layer-num', type=int, default=2)
parser.add_argument('--norm-type', type=str, default='post')  # 'post' or 'pre'
parser.add_argument("--bl-log-sig-min", type=float, default=-20)
parser.add_argument("--bl-log-sig-max", type=float, default=2)
parser.add_argument("--lr-bl", type=float, default=0.00001, help="learning rate")

parser.add_argument("--frame-per-proc", type=int, default=128, help="number of frames per process before update")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
	env = utils.make_env(args.env, args.seed + 10000 * i)
	envs.append(env)
env = ParallelEnv(envs)
print("Environments loaded\n")

# Load agent
# model_dir = utils.get_model_dir(args.model)
# print("args.model", args.model) # CrossingS11N5

model_dir = utils.get_model_dir("block/" + args.model + "/lr_bl=" + str(args.lr_bl) + "/L=" + str(args.block_length) + "_k=" + str(args.top_k)
                                + "/" + str(args.seed))

agent = utils.Agent_Block(env.observation_space, env.action_space, model_dir,
					argmax=args.argmax, num_envs=args.procs,
					use_memory=args.memory, use_text=args.text, frame_per_proc=args.frame_per_proc,
					block_length=args.block_length, top_k=args.top_k, block_hid_dim=args.block_hid_dim, att_layer_num=args.att_layer_num,
					norm_type=args.norm_type, bl_log_sig_min=args.bl_log_sig_min, bl_log_sig_max=args.bl_log_sig_max
						  )
print("Agent loaded\n")

# Initialize logs

logs = {"num_frames_per_episode": [], "return_per_episode": []}

# Run agent

start_time = time.time()

obss = env.reset()

log_done_counter = 0
log_episode_return = torch.zeros(args.procs, device=device)
log_episode_num_frames = torch.zeros(args.procs, device=device)

## Newly added
periodic_counter = 0
prev_dones = tuple([False for _ in range(args.procs)])

while log_done_counter < args.episodes:
	actions, cnn_memory = agent.get_actions(obss)
	obss, rewards, dones, _ = env.step(actions)
	agent.analyze_feedbacks(rewards, prev_dones, dones, periodic_counter, cnn_memory, device=device)
	prev_dones = dones

	periodic_counter += 1
	if periodic_counter % args.frame_per_proc == 0:
		periodic_counter = 0

	log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
	log_episode_num_frames += torch.ones(args.procs, device=device)

	for i, done in enumerate(dones):
		if done:
			log_done_counter += 1
			logs["return_per_episode"].append(log_episode_return[i].item())
			logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

	mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
	log_episode_return *= mask
	log_episode_num_frames *= mask

end_time = time.time()

# Print logs

num_frames = sum(logs["num_frames_per_episode"])
fps = num_frames/(end_time - start_time)
duration = int(end_time - start_time)
return_per_episode = utils.synthesize(logs["return_per_episode"])
num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
	  .format(num_frames, fps, duration,
			  *return_per_episode.values(),
			  *num_frames_per_episode.values()))