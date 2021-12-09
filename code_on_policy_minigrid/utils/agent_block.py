import torch

import utils
from .other import device
# from model import ACModel
from model_block import ACModel_Block, BLModel_Block


class Agent_Block:
	"""An agent.

	It is able:
	- to choose an action given an observation,
	- to analyze the feedback (i.e. reward and done state) of its action."""

	def __init__(self, obs_space, action_space, model_dir,
				 argmax=False, num_envs=1, use_memory=False, use_text=False, frame_per_proc=128,
				 block_length=8, top_k=4, block_hid_dim=256, att_layer_num=2,
				 norm_type='post', bl_log_sig_min=-20, bl_log_sig_max=2):
		obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
		self.acmodel = ACModel_Block(obs_space, action_space, use_memory=use_memory, use_text=use_text)
		self.argmax = argmax
		self.num_envs = num_envs

		if self.acmodel.recurrent:
			self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

		self.acmodel.load_state_dict(utils.get_model_state(model_dir))
		self.acmodel.to(device)
		self.acmodel.eval()
		if hasattr(self.preprocess_obss, "vocab"):
			self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))
			
		## Block model
		self.blmodel = BLModel_Block(obs_space, top_k, block_hid_dim, att_layer_num, norm_type, device,
									 bl_log_sig_min, bl_log_sig_max)

		self.block_length = block_length
		self.frame_per_proc = frame_per_proc #128
		
		self.block_memory = torch.zeros(self.num_envs, self.blmodel.block_hid_dim, device=device)
		self.cnn_memories_post = torch.zeros(self.block_length + self.frame_per_proc,
											 self.num_envs, self.acmodel.image_embedding_size, device=device)
		self.block_counter = torch.zeros(self.num_envs, device=device, dtype=torch.int)
		
		self.blmodel.load_state_dict(utils.get_bl_model_state(model_dir))
		self.blmodel.to(device)
		self.blmodel.eval()

		with torch.no_grad():
			self.mu_sig_memory = self.blmodel.block_mu_sig(block_memory=self.block_memory)


	def get_actions(self, obss):
		preprocessed_obss = self.preprocess_obss(obss, device=device)
		
		with torch.no_grad():
			dist, _, self.memories, cnn_memory = self.acmodel(preprocessed_obss, self.memories, self.mu_sig_memory)
			# if self.acmodel.recurrent:
			# 	dist, _, self.memories, cnn_memory = self.acmodel(preprocessed_obss, self.memories, self.mu_sig_memory)
			# else:
			# 	dist, _ = self.acmodel(preprocessed_obss)

		# Original
		# with torch.no_grad():
		# 	if self.acmodel.recurrent:
		# 		dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
		# 	else:
		# 		dist, _ = self.acmodel(preprocessed_obss)

		if self.argmax:
			actions = dist.probs.max(1, keepdim=True)[1]
		else:
			actions = dist.sample()

		return actions.cpu().numpy(), cnn_memory

	# def get_action(self, obs):
	# 	return self.get_actions([obs])[0]

	def analyze_feedbacks(self, rewards, prev_dones, dones, i, cnn_memory, device):
		if self.acmodel.recurrent:
			next_mask = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
			self.memories *= next_mask
			
			### Update block latent variables h and its corresponding mu, sig
			next_mask_sq = 1 - torch.tensor(dones, dtype=torch.float, device=device)
			
			### CNN inputs
			self.cnn_memories_post[i + self.block_length] = cnn_memory
			### Block
			# self.mu_sig_memories[i] = self.mu_sig_memory  ## self.mu_sig_memory.requires_grad True
			
			### Saving Block, Gradient is not needed
			with torch.no_grad():
				### New block variable when activated
				bl_update_check = (self.block_counter % self.block_length == self.block_length - 1)  # length 16=batch. True or False
				termination_check = torch.tensor(dones, device=device)  # length 16=batch. True or False
				double_check = bl_update_check + termination_check  # OR operation
				
				if ~torch.all(~double_check):  ## At least 1 True in double_check (termination or block length == L-1).
					# self.block_counter_mask[i + self.block_length] = double_check
					
					termination_only_check = termination_check * (~bl_update_check)  # termination \ bl_update: termination only
					mask = 1 - torch.tensor(prev_dones, dtype=torch.float, device=device).unsqueeze(1)
					
					## Case 1
					if ~torch.all(~bl_update_check):  ## At least 1 True in bl_update_check. ~: complement
						## Do need self.block_memory * self.mask.unsqueeze(1) for initialization for h_n.
						self.block_memory, _ = self.blmodel(self.cnn_memories_post[i + 1: i + self.block_length + 1],
						                                    self.block_memory * mask, bl_update_check)
						
					## Case 2
					if ~torch.all(~termination_only_check):  ## Caution! elif(X), if(O). Case 1 and 2 can happen at the same step
						remained_length = self.block_counter % self.block_length  # Each element is 0 <= element <= L-1.
						termination_only_index = termination_only_check.nonzero()
						
						## Do need self.block_memory * self.mask.unsqueeze(1) for initialization for h_n.
						for itm in termination_only_index:  # 'itm' contains only one index # itm.item()
							remained_length_itm = remained_length[itm].item()
							## Remained length is 1 <= 1 + remained_length_itm < L.
							self.block_memory, _ = self.blmodel(self.cnn_memories_post[i - remained_length_itm + self.block_length: i + self.block_length + 1],
							                                    self.block_memory * mask, itm)
					
					### Update mu_sig after finishing update of block_memory
					## Do need self.block_memory * next_mask.unsqueeze(1) for initialization for h_n.
					self.mu_sig_memory = self.blmodel.block_mu_sig(block_memory=self.block_memory * next_mask)
				
				### block variable counter
				self.block_counter = (self.block_counter + 1) * next_mask_sq.to(device=device, dtype=torch.int)  ## integer!
				# self.block_counter = (self.block_counter + 1) * next_mask_sq.type(torch.IntTensor) ## integer!
				
			### CNN memory update. requires_grad=False
			if i % self.frame_per_proc == self.frame_per_proc - 1:
				self.cnn_memories_post = torch.cat((self.cnn_memories_post[-self.block_length:],
				                                    torch.zeros(self.frame_per_proc, self.num_envs, self.acmodel.image_embedding_size,
				                                                device=device)), dim=0)  # [L+128, 16, 128]
		
	# def analyze_feedback(self, reward, done):
	# 	return self.analyze_feedbacks([reward], [done])
