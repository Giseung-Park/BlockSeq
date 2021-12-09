from abc import ABC, abstractmethod
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv

import numpy as np
from torch import distributions as dis


class BaseAlgo_Block(ABC):
	"""The base class for RL algorithms."""

	def __init__(self, envs, acmodel, blmodel, device, num_frames_per_proc, discount, lr, lr_bl, gae_lambda, entropy_coef,
				 value_loss_coef, max_grad_norm, bl_max_grad_norm, recurrence, preprocess_obss, reshape_reward, block_length, epochs_bl,
				 self_norm_sample):
		"""
		Initializes a `BaseAlgo_Block` instance.

		Parameters:
		----------
		envs : list
			a list of environments that will be run in parallel
		acmodel : torch.Module
			the model
		num_frames_per_proc : int
			the number of frames collected by every process for an update
		discount : float
			the discount for future rewards
		lr : float
			the learning rate for optimizers
		gae_lambda : float
			the lambda coefficient in the GAE formula
			([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
		entropy_coef : float
			the weight of the entropy cost in the final objective
		value_loss_coef : float
			the weight of the value loss in the final objective
		max_grad_norm : float
			gradient will be clipped to be at most this value
		recurrence : int
			the number of steps the gradient is propagated back in time
		preprocess_obss : function
			a function that takes observations returned by the environment
			and converts them into the format that the model can handle
		reshape_reward : function
			a function that shapes the reward, takes an
			(observation, action, reward, done) tuple as an input
		"""

		# Store parameters

		self.env = ParallelEnv(envs)
		self.acmodel = acmodel
		
		## Block model
		self.blmodel = blmodel
		self.block_length = block_length
		self.lr_bl = lr_bl
		self.epochs_bl = epochs_bl
		self.self_norm_sample = self_norm_sample
		
		self.device = device
		self.num_frames_per_proc = num_frames_per_proc
		self.discount = discount
		self.lr = lr
		self.gae_lambda = gae_lambda
		self.entropy_coef = entropy_coef
		self.value_loss_coef = value_loss_coef
		self.max_grad_norm = max_grad_norm
		self.bl_max_grad_norm = bl_max_grad_norm ## block grad clipping
		self.recurrence = recurrence
		self.preprocess_obss = preprocess_obss or default_preprocess_obss
		self.reshape_reward = reshape_reward

		# Control parameters

		assert self.acmodel.recurrent or self.recurrence == 1
		assert self.num_frames_per_proc % self.recurrence == 0

		# Configure acmodel

		self.acmodel.to(self.device)
		self.acmodel.train()
		
		self.blmodel.to(self.device)
		self.blmodel.eval() ## Initial setting; turn off drop-out in self-attention.

		# Store helpers values

		self.num_procs = len(envs)
		self.num_frames = self.num_frames_per_proc * self.num_procs

		# Initialize experience values

		shape = (self.num_frames_per_proc, self.num_procs)
		extended_shape = (self.block_length + self.num_frames_per_proc, self.num_procs)
		
		self.shape = shape

		self.obs = self.env.reset()
		self.obss = [None]*(shape[0])
		if self.acmodel.recurrent:
			self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
			self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
			### Block memory
			## When we use \tilde{h_n}
			self.block_memory = torch.zeros(shape[1], self.blmodel.block_hid_dim, device=self.device)
			self.block_memory_inference = torch.zeros(shape[1], self.blmodel.block_hid_dim, device=self.device)
			
			## Used Only for model learning
			## 1) block_memories
			self.remained_idx_inference = [0 for _ in range(shape[1])] # [16]
			## 2) Save All the Y_n^k for model learning. [128, 16, top_k*64]
			# self.selected_elt_memories = torch.zeros(*shape, self.blmodel.image_embedding_size*self.blmodel.top_k, device=self.device)
			
			## When we use mu and sigma
			with torch.no_grad():
				self.mu_sig_memory = self.blmodel.block_mu_sig(block_memory=self.block_memory) # [16, 64*2], grad=T
			# self.mu_sig_memories = torch.zeros(*shape, self.blmodel.block_mu_size*2, device=self.device) # [128, 16, 64*2], grad=F
			# print("Init", self.mu_sig_memory.shape, self.mu_sig_memory.requires_grad)
			# print("Init 2", self.mu_sig_memories.shape, self.mu_sig_memories.requires_grad)
			
			### CNN inputs memory
			self.cnn_memories_post = torch.zeros(*extended_shape, self.acmodel.image_embedding_size, device=self.device)
			## batchwise counter for block variable generation
			self.block_counter = torch.zeros(shape[1], device=self.device, dtype=torch.int)
			# self.block_counter_mask = torch.zeros(*extended_shape, device=self.device, dtype=torch.bool)
			# self.block_counter_mask[self.block_length-1] = torch.ones(extended_shape[1], device=self.device, dtype=torch.bool)
			
			## For assertion test
			self.termination_counter_mask = torch.zeros(*shape, device=self.device, dtype=torch.bool)
			
			### Block list for model learning
			self.block_list = [[] for _ in range(self.num_procs)]
			self.block_done_list = [[False] for _ in range(self.num_procs)]
			
			## block model optimizer
			## lr: tuning is needed. e.g., block_lr
			self.block_optimizer = torch.optim.Adam(self.blmodel.parameters(), self.lr_bl)
			
			
		self.mask = torch.ones(shape[1], device=self.device)
		self.masks = torch.zeros(*shape, device=self.device)
		self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
		self.values = torch.zeros(*shape, device=self.device)
		self.rewards = torch.zeros(*shape, device=self.device)
		self.advantages = torch.zeros(*shape, device=self.device)
		self.log_probs = torch.zeros(*shape, device=self.device)
		
		
		# Initialize log values

		self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
		self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
		self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

		self.log_done_counter = 0
		self.log_return = [0] * self.num_procs
		self.log_reshaped_return = [0] * self.num_procs
		self.log_num_frames = [0] * self.num_procs

	def collect_experiences(self):
		"""Collects rollouts and computes advantages.

		Runs several environments concurrently. The next actions are computed
		in a batch mode for all environments at the same time. The rollouts
		and advantages from all environments are concatenated together.

		Returns
		-------
		exps : DictList
			Contains actions, rewards, advantages etc as attributes.
			Each attribute, e.g. `exps.reward` has a shape
			(self.num_frames_per_proc * num_envs, ...). k-th block
			of consecutive `self.num_frames_per_proc` frames contains
			data obtained from the k-th environment. Be careful not to mix
			data from different environments!
		logs : dict
			Useful stats about the training process, including the average
			reward, policy loss, value loss, etc.
		"""
		
		# print("check state", self.blmodel.training) # False: eval
		self.termination_counter_mask = torch.zeros(*self.termination_counter_mask.shape, device=self.device, dtype=torch.bool)

		for i in range(self.num_frames_per_proc):
			# Do one agent-environment interaction

			## We do not need self.mu_sig_memory * self.mask.unsqueeze(1) as we already processed masking in h_n.
			preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
			with torch.no_grad():
				if self.acmodel.recurrent:
					dist, value, memory, cnn_memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1),
													   self.mu_sig_memory)
				else:
					dist, value = self.acmodel(preprocessed_obs)
			action = dist.sample()
			
			obs, reward, done, _ = self.env.step(action.cpu().numpy())
			
			# Update experiences values

			self.obss[i] = self.obs
			self.obs = obs
			next_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
			
			if self.acmodel.recurrent:
				self.memories[i] = self.memory
				self.memory = memory
				### CNN inputs
				self.cnn_memories_post[i + self.block_length] = cnn_memory
				### Block
				# self.mu_sig_memories[i] = self.mu_sig_memory  ## self.mu_sig_memory.requires_grad True
				
				### Saving Block, Gradient is not needed
				with torch.no_grad():
					### New block variable when activated
					bl_update_check = (self.block_counter % self.block_length == self.block_length - 1) # length 16=batch. True or False
					termination_check = torch.tensor(done, device=self.device) # length 16=batch. True or False
					double_check = bl_update_check + termination_check # OR operation
	
					if ~torch.all(~double_check):  ## At least 1 True in double_check (termination or block length == L-1).
						# self.block_counter_mask[i + self.block_length] = double_check
						
						self.termination_counter_mask[i] = termination_check
						termination_only_check = termination_check * (~bl_update_check)  # termination \ bl_update: termination only
	
						## Case 1
						if ~torch.all(~bl_update_check):  ## At least 1 True in bl_update_check. ~: complement
							## Do need self.block_memory * self.mask.unsqueeze(1) for initialization for h_n.
							self.block_memory, _ = self.blmodel(self.cnn_memories_post[i + 1: i + self.block_length + 1],
							                                    self.block_memory * self.mask.unsqueeze(1), bl_update_check)
							
							bl_update_check_index = bl_update_check.nonzero()
							not_termination_check_index = (~termination_check).nonzero()
							
							for itm in bl_update_check_index:  # (seq, batch, dim) = (self.block_length, 1, dim)
								idx = itm.item()
								self.block_list[idx].append(self.cnn_memories_post[i + 1: i + self.block_length + 1, itm, :])
								if itm in not_termination_check_index:
									self.block_done_list[idx].append(False)
								else:
									self.block_done_list[idx].append(True)
							
						## Case 2
						if ~torch.all(~termination_only_check): ## Caution! elif(X), if(O). Case 1 and 2 can happen at the same step
							remained_length = self.block_counter % self.block_length  # Each element is 0 <= element <= L-1.
							termination_only_index = termination_only_check.nonzero()
							
							## Do need self.block_memory * self.mask.unsqueeze(1) for initialization for h_n.
							for itm in termination_only_index:  # 'itm' contains only one index # itm.item()
								idx = itm.item()
								remained_length_itm = remained_length[itm].item()
								## Remained length is 1 <= 1 + remained_length_itm < L.
								self.block_memory, _ = self.blmodel(self.cnn_memories_post[
									i - remained_length_itm + self.block_length: i + self.block_length + 1],
									self.block_memory * self.mask.unsqueeze(1), itm)
								
								self.block_list[idx].append(self.cnn_memories_post
								[i - remained_length_itm + self.block_length: i + self.block_length + 1, itm, :])
								self.block_done_list[idx].append(True)  # (seq, batch, dim) = (1 + remained_length_itm, 1, dim)
						
						### Update mu_sig after finishing update of block_memory
						## Do need self.block_memory * next_mask.unsqueeze(1) for initialization for h_n.
						self.mu_sig_memory = self.blmodel.block_mu_sig(block_memory=self.block_memory*next_mask.unsqueeze(1))
						
					### block variable counter
					self.block_counter = (self.block_counter + 1) * next_mask.to(device=self.device, dtype=torch.int)  ## integer!
					# self.block_counter = (self.block_counter + 1) * next_mask.type(torch.IntTensor) ## integer!

			self.masks[i] = self.mask
			# self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
			self.mask = next_mask
			self.actions[i] = action
			self.values[i] = value
			if self.reshape_reward is not None:
				self.rewards[i] = torch.tensor([
					self.reshape_reward(obs_, action_, reward_, done_)
					for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
				], device=self.device)
			else:
				self.rewards[i] = torch.tensor(reward, device=self.device)
			self.log_probs[i] = dist.log_prob(action)
			
			# Update log values

			self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float) # Tuple
			self.log_episode_reshaped_return += self.rewards[i]
			self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device) # Tensor

			for i, done_ in enumerate(done):
				if done_:
					self.log_done_counter += 1
					self.log_return.append(self.log_episode_return[i].item())
					self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
					self.log_num_frames.append(self.log_episode_num_frames[i].item())

			self.log_episode_return *= self.mask
			self.log_episode_reshaped_return *= self.mask
			self.log_episode_num_frames *= self.mask

		# Add advantage and return to experiences

		preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
		with torch.no_grad():
			if self.acmodel.recurrent:
				_, next_value, _, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), self.mu_sig_memory)
			else:
				_, next_value = self.acmodel(preprocessed_obs)

		for i in reversed(range(self.num_frames_per_proc)):
			next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
			next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
			next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

			delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
			self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

		# Define experiences:
		#   the whole experience is the concatenation of the experience
		#   of each process.
		# In comments below:
		#   - T is self.num_frames_per_proc,
		#   - P is self.num_procs,
		#   - D is the dimensionality.
		
		exps = DictList()
		exps.obs = [self.obss[i][j]
					for j in range(self.num_procs)
					for i in range(self.num_frames_per_proc)]
		if self.acmodel.recurrent:
			# T x P x D -> P x T x D -> (P * T) x D
			exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
			# T x P -> P x T -> (P * T) x 1
			exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
		# for all tensors below, T x P -> P x T -> P * T
		exps.action = self.actions.transpose(0, 1).reshape(-1)
		exps.value = self.values.transpose(0, 1).reshape(-1)
		exps.reward = self.rewards.transpose(0, 1).reshape(-1)
		exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
		exps.returnn = exps.value + exps.advantage
		exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

		# Preprocess experiences
		exps.obs = self.preprocess_obss(exps.obs, device=self.device)
		
		# assert [sum(lit) for lit in self.block_done_list] == torch.sum(self.termination_counter_mask, dim=0).tolist()
		assert [sum(lit[1:]) for lit in self.block_done_list] == torch.sum(self.termination_counter_mask, dim=0).tolist()
		
		### CNN memory update. requires_grad=False
		self.cnn_memories_post = torch.cat((self.cnn_memories_post[-self.block_length:],
			torch.zeros(*self.shape, self.acmodel.image_embedding_size, device=self.device)), dim=0)  # [L+128, 16, 128]
			
			
		## I. model learning (asynchronous)
		self.blmodel.train()
		
		for epoch_idx in range(self.epochs_bl):
			indexes = np.arange(0, self.num_procs)
			indexes = np.random.permutation(indexes)
			
			
			for idx in indexes:
				loss_p = torch.zeros(1, device=self.device)[0]
				loss_q = torch.zeros(1, device=self.device)[0]
				
				block_memory = torch.zeros(1, self.blmodel.block_hid_dim, device=self.device) ## Memory Initialize
				block_done_list_fl = 1 - torch.tensor(self.block_done_list[idx][:-1], device=self.device, dtype=torch.float)
				
				for block, block_done in zip(self.block_list[idx], block_done_list_fl):
					# print("self.blmodel.parameters()", [item for item in self.blmodel.parameters()])
					block_memory, q_output_select = self.blmodel(block, block_memory*block_done, [0]) # [1,256]; [1,64*top_k]
					
					### Model learning part
					mu_sig_tensor = self.blmodel.block_mu_sig(block_memory) # [1,64*2]
					
					# print("block_memory", block_memory)
					# print("mu", mu_sig_tensor[:, :self.blmodel.block_mu_size], mu_sig_tensor[:, :self.blmodel.block_mu_size].shape)
					# print("sig", mu_sig_tensor[:, self.blmodel.block_mu_size:], mu_sig_tensor[:, self.blmodel.block_mu_size:].shape)
					# print()
					
					block_q_prob = dis.Normal(mu_sig_tensor[:, :self.blmodel.block_mu_size], mu_sig_tensor[:, self.blmodel.block_mu_size:]) # [1, 64]
					block_variable_samples = block_q_prob.rsample(sample_shape=[self.self_norm_sample])  # [self_norm_sample, 1, 64]
					
					# print("log_prob", block_q_prob.log_prob(block_variable_samples))
					logit_q = torch.sum(block_q_prob.log_prob(block_variable_samples), dim=-1).permute(1, 0)  # [1, self_norm_sample]
					# logit_q = torch.sum(block_q_prob.log_prob(block_variable_samples).clamp(-20, 10), dim=-1).permute(1,0) # [1, self_norm_sample]
					
					##### original
					q_output_select_tile = q_output_select.detach().unsqueeze(dim=0).repeat(self.self_norm_sample, 1, 1)  # [self_norm_sample, 1, 64*top_k]
					# q_output_select_tile = q_output_select.unsqueeze(dim=0).repeat(self_norm_sample, 1, 1) # [self_norm_sample, 1, 64*top_k]
					
					logit_p = self.blmodel.self_norm_model(
						torch.cat((q_output_select_tile, block_variable_samples.detach()), dim=-1)).squeeze(dim=-1).permute(
						1, 0)  # [1, self_norm_sample]
					
					### We now have logit_p tensor w.r.t. theta, and logit_q tensor w.r.t. phi.

					### !!! Important! Weight should be sampled 'values', so we use .detach().
					log_weight = (logit_p - logit_q).detach()
					
					log_weight_max, _ = torch.max(log_weight, dim=-1, keepdim=True)  # [batch, 1]
					### leave log_weight_amax for later ablation study.
					# print("log_weight - log_weight_max", log_weight - log_weight_max) # [batch, self_norm_sample] by broadcasting
					# print("sum", torch.sum( torch.exp( log_weight - log_weight_max ), dim=-1, keepdim=True  )) # [batch, 1]

					log_sum_weight = log_weight_max + torch.log(
						torch.sum(torch.exp(log_weight - log_weight_max), dim=-1, keepdim=True))  # [batch, 1]
					# print("log_sum_weight", log_sum_weight, log_sum_weight.shape)

					self_normalized_weight = torch.exp(log_weight - log_sum_weight)  # [batch, self_norm_sample] by broadcasting

					### detach check
					# self_normalized_weight = self_normalized_weight.detach()
					# ##### Step 2. Calculate 'loss' log p_theta and log q_phi.
					# print("logit_p", logit_p, logit_p.shape) # [batch=4, sample_num_K]
					# print("logit_q", logit_q, logit_q.shape)
					# if logit_p.isnan().any() or logit_q.isnan().any():
					# 	print("NaN!!!!!!!!!!!!!!!!!!!!!!")

					weighted_logit_p = self_normalized_weight * logit_p
					weighted_logit_q = self_normalized_weight * logit_q

					# print("inner size", torch.mean(torch.sum(weighted_logit_p, dim=-1)), torch.mean(torch.sum(weighted_logit_p, dim=-1)).shape)

					loss_p += torch.mean(torch.sum(weighted_logit_p, dim=-1))
					loss_q += torch.mean(torch.sum(weighted_logit_q, dim=-1))
					
					## OR
					# loss_p += torch.sum(torch.sum(weighted_logit_p, dim=-1))
					# loss_q += torch.sum(torch.sum(weighted_logit_q, dim=-1))
				
				loss_total = -(loss_p + loss_q)
				# loss_total = -(loss_p + 10*loss_q)
				
				self.block_optimizer.zero_grad()
				loss_total.backward()
				# bl_grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.blmodel.parameters()) ** 0.5
				# print("bl_grad_norm", bl_grad_norm)
				torch.nn.utils.clip_grad_norm_(self.blmodel.parameters(), self.bl_max_grad_norm)
				self.block_optimizer.step()
	
				# check original loss to maximize.
				# if epoch_idx == self.epochs_bl - 1 and idx == indexes[-1]:
				# 	print("loss_p", loss_p)
				# 	print("loss_q", loss_q)
				# 	print("~~~~~~~~~~~~~~~~~~")
				# 	print()
				
		### Model learning ends
		
		### Regenerating block_memory and mu_sig_memory
		self.blmodel.eval()
		
		with torch.no_grad(): ## Reduce memory
			self.block_memories = torch.zeros(*(self.num_frames_per_proc, self.num_procs, self.blmodel.block_hid_dim), device=self.device)
			
			for idx in range(self.num_procs):
				count = 0
				block_done_list_fl = 1 - torch.tensor(self.block_done_list[idx], device=self.device, dtype=torch.float)
				
				for block, block_done in zip(self.block_list[idx], block_done_list_fl[:-1]):
					bl_length = block.shape[0] - self.remained_idx_inference[idx]
					block_memory_inference_w_done = self.block_memory_inference[[idx]] * block_done
					
					self.block_memories[count:count+bl_length, [idx], :] = block_memory_inference_w_done.unsqueeze(dim=0).repeat(bl_length,1,1)
					
					self.block_memory_inference[[idx]], _ \
						= self.blmodel(block, block_memory_inference_w_done, [0])  # [1,256]; [1,64*top_k]
					count += bl_length
					if self.remained_idx_inference[idx] > 0:
						self.remained_idx_inference[idx] = 0
					
				if count < self.num_frames_per_proc:
					add_bl_length = self.num_frames_per_proc - count
					block_memory_inference_w_done = self.block_memory_inference[[idx]] * block_done_list_fl[-1]
					
					self.block_memories[count:count + add_bl_length, [idx], :] \
						= block_memory_inference_w_done.unsqueeze(dim=0).repeat(add_bl_length, 1, 1)
					count += add_bl_length
					self.remained_idx_inference[idx] = add_bl_length
				
				assert count == self.num_frames_per_proc # 128
				
			# print("block memory shape", self.block_memories, self.block_memories.requires_grad) # True
			# print(self.memories.requires_grad, self.cnn_memories_post.requires_grad, self.block_memories.requires_grad) # F F T
	
			## mu, sig tensors for RL learning, detach()
			block_mu_sig = self.blmodel.block_mu_sig(block_memory=self.block_memories) # [128, 16, 64*2] each. grad=F
			# print("block_mu_sig", block_mu_sig.requires_grad)
		
		exps.mu_sig_memory = block_mu_sig.transpose(0, 1).reshape(-1, *block_mu_sig.shape[2:])
		del self.block_memories
		
		### Important! Re-initialize for the next 128 procedures
		self.block_list = [[] for _ in range(self.num_procs)]
		self.block_done_list = [[bl_done[-1]] for bl_done in self.block_done_list]

		# Log some values

		keep = max(self.log_done_counter, self.num_procs)

		logs = {
			"return_per_episode": self.log_return[-keep:],
			"reshaped_return_per_episode": self.log_reshaped_return[-keep:],
			"num_frames_per_episode": self.log_num_frames[-keep:],
			"num_frames": self.num_frames
		}
		
		self.log_done_counter = 0
		self.log_return = self.log_return[-self.num_procs:]
		self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
		self.log_num_frames = self.log_num_frames[-self.num_procs:]

		return exps, logs

	@abstractmethod
	def update_parameters(self):
		pass
