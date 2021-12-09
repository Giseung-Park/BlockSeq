import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys

import utils
from utils import device
from model_block import ACModel_Block, BLModel_Block


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
					help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
					help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
					help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--log-interval", type=int, default=1,
					help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=20,
					help="number of updates between two saves (default: 20, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
					help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
					help="number of frames of training (default: 1e7)")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
					help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
					help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
					help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
					help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
					help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
					help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
					help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
					help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
					help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
					help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
					help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
					help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--text", action="store_true", default=False,
					help="add a GRU to the model to handle text input")

parser.add_argument("--recurrence", type=int, default=32,
					help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")

### What to change
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)") #1-5

## Parameters related to block learning / variable
parser.add_argument("--block-length", type=int, default=8, help="Length L of each block B_n (L > 1)")
parser.add_argument("--top-k", type=int, default=4, help="Number of k selected elements in B_n. 1 <= k <= L.")
parser.add_argument("--lr-bl", type=float, default=0.00001, help="learning rate")
parser.add_argument("--epochs-bl", type=int, default=2, help="number of epochs for block model learning (default: 2)")
parser.add_argument("--self-norm-sample", type=int, default=50, help="number of samples for self-normalized IS (default: 50)")
parser.add_argument('--block-hid-dim', type=int, default=256)
parser.add_argument('--att-layer-num', type=int, default=2)
parser.add_argument('--norm-type', type=str, default='post') # 'post' or 'pre'
parser.add_argument("--bl-max-grad-norm", type=float, default=0.1, help="maximum norm of block gradient (default: 0.1)")
parser.add_argument("--bl-log-sig-min", type=float, default=-20)
parser.add_argument("--bl-log-sig-max", type=float, default=2)


args = parser.parse_args()

assert args.top_k <= args.block_length

args.mem = args.recurrence > 1

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir("block/"+ model_name + "/lr_bl=" + str(args.lr_bl) + "/L=" + str(args.block_length) + "_k=" + str(args.top_k)
								 + "/" + str(args.seed))

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
	envs.append(utils.make_env(args.env, args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Load training status

try:
	status = utils.get_status(model_dir)
except OSError:
	status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
	preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model

acmodel = ACModel_Block(obs_space, envs[0].action_space, args.mem, args.text)
if "model_state" in status:
	acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Stepwise Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

## Load block model
blmodel = BLModel_Block(obs_space, args.top_k, args.block_hid_dim, args.att_layer_num, args.norm_type, device,
						args.bl_log_sig_min, args.bl_log_sig_max)
if "bl_model_state" in status:
	blmodel.load_state_dict(status["bl_model_state"])
blmodel.to(device)
txt_logger.info("Blockwise Model loaded\n")
txt_logger.info("{}\n".format(blmodel))

# Load algo

if args.algo == "a2c":
	raise NotImplementedError

elif args.algo == "ppo":
	algo = torch_ac.PPOAlgo_Block(envs, acmodel, blmodel, device, args.frames_per_proc, args.discount, args.lr, args.lr_bl,
							args.gae_lambda,
							args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.bl_max_grad_norm, args.recurrence,
							args.optim_eps, args.clip_eps, args.epochs, args.epochs_bl, args.batch_size, preprocess_obss,
								  block_length=args.block_length, self_norm_sample=args.self_norm_sample)
else:
	raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if "optimizer_state" in status:
	algo.optimizer.load_state_dict(status["optimizer_state"])
if "bl_optimizer_state" in status:
	algo.block_optimizer.load_state_dict(status["bl_optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
	# Update model parameters

	update_start_time = time.time()
	# print("collect")
	exps, logs1 = algo.collect_experiences()
	# print("update")
	logs2 = algo.update_parameters(exps)
	logs = {**logs1, **logs2}
	update_end_time = time.time()

	num_frames += logs["num_frames"]
	update += 1

	# Print logs

	if update % args.log_interval == 0:
		fps = logs["num_frames"]/(update_end_time - update_start_time)
		duration = int(time.time() - start_time)
		return_per_episode = utils.synthesize(logs["return_per_episode"])
		rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
		num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

		header = ["update", "frames", "FPS", "duration"]
		data = [update, num_frames, fps, duration]
		header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
		data += rreturn_per_episode.values()
		header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
		data += num_frames_per_episode.values()
		header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
		data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

		txt_logger.info(
			"U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
			.format(*data))

		header += ["return_" + key for key in return_per_episode.keys()]
		data += return_per_episode.values()

		if status["num_frames"] == 0:
			csv_logger.writerow(header)
		csv_logger.writerow(data)
		csv_file.flush()

		for field, value in zip(header, data):
			tb_writer.add_scalar(field, value, num_frames)

	# Save status

	if args.save_interval > 0 and update % args.save_interval == 0:
		status = {"num_frames": num_frames, "update": update,
				  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict(),
				  "bl_model_state": blmodel.state_dict(), "bl_optimizer_state": algo.block_optimizer.state_dict()}
		if hasattr(preprocess_obss, "vocab"):
			status["vocab"] = preprocess_obss.vocab.vocab
		utils.save_status(status, model_dir)
		txt_logger.info("Status saved")
