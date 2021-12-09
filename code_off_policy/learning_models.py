import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torch import distributions as dis

from attention_model import BaseModel

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
REG = 1e-3  # regularization of the actor
SIG_MIN = 1e-3

class Block_model(nn.Module):
	def __init__(self,
	             trans_model_q: BaseModel,
	             input_size,
	             action_size,
	             rnn_type='mtgru',  # 'mtlstm'
	             d_layers=[256],
	             z_layers=[64],
	             taus=[1.0, ],
	             x_phi_layers=[128, ],
	             posterior_layers=[128, ],
	             prior_layers=[128, ],
	             lr_st=8e-4, # 8e-4
	             optimizer='adam',
	             predict_done=False,
	             feedforward_actfun_rnn=nn.Tanh,
	             sig_scale='auto',
	             sample_num_K=50,
	             top_k=1):

		super(Block_model, self).__init__()
		
		if len(d_layers) != len(taus):
			raise ValueError("Length of hidden layer size and timescales should be the same.")
		
		# Network layer parameters
		self.input_size = input_size
		self.action_size = action_size  ## 2
		
		self.d_layers = d_layers
		self.z_layers = z_layers
		self.taus = taus
		self.rnn_type = rnn_type
		self.n_levels = len(d_layers)
		self.x_phi_layers = x_phi_layers  # feature-extracting transformations
		self.prior_layers = prior_layers
		self.posterior_layers = posterior_layers
		self.action_feedback = False  # Original: True
		self.batch = True
		self.predict_done = predict_done
		self.sig_scale = sig_scale
		
		
		### Newly added for Bi-directional Attention
		self.K = sample_num_K
		self.top_k = top_k
		
		self.trans_model_q = trans_model_q
		# self.trans_select_N = 2 # 2 as bi-LSTM. Can be changed
		
		# feature-extracting transformations
		x2phi = nn.ModuleList()
		last_layer_size = self.input_size
		for layer_size in self.d_layers:
			x2phi.append(nn.Linear(last_layer_size, layer_size, bias=True))
			# last_layer_size = 2*layer_size
			last_layer_size = layer_size
			x2phi.append(feedforward_actfun_rnn())
		x2phi.append(nn.Linear(last_layer_size, self.d_layers[0] - self.action_size, bias=True))  ### last layer: 128->256
		
		self.f_x2phi = nn.Sequential(*x2phi)
		
		# input encoding layers
		# self.xphi2h0 = nn.Linear(self.x_phi_layers[-1], self.d_layers[0], bias=True)
		
		### Newly added
		rws_trans = nn.ModuleList()
		rws_trans.append(nn.Linear(self.top_k*self.d_layers[0] + self.z_layers[0], self.d_layers[0], bias=True))
		rws_trans.append(feedforward_actfun_rnn())
		
		# rws_trans = nn.ModuleList()
		# rws_trans.append(nn.Linear(self.d_layers[0] + self.z_layers[0], self.d_layers[0], bias=True))
		# rws_trans.append(feedforward_actfun_rnn())
		# rws_trans.append(nn.ReLU())
		
		# rws_trans.append(nn.Linear(self.d_layers[0], self.d_layers[0], bias=True))
		# rws_trans.append(feedforward_actfun_rnn())
		# rws_trans.append(nn.ReLU())
		
		rws_trans.append(nn.Linear(self.d_layers[0], 1, bias=True))
		self.rws_logit = nn.Sequential(*rws_trans)
		# self.rws_logit = nn.Linear(self.trans_select_N*self.d_layers[0] + self.z_layers[0], 1, bias=True)
		
		## block variable sample
		self.f_dphi2mu_q_block = nn.ModuleList()
		if isinstance(self.sig_scale, float):
			self.f_dphi2sig_q_block = lambda x: torch.tensor(self.sig_scale, dtype=torch.float32)
		# self.f_d2sig_p = lambda x: torch.tensor(self.sig_scale, dtype=torch.float32)
		else:
			self.f_dphi2sig_q_block = nn.ModuleList()
		# self.f_d2sig_p = nn.ModuleList()
		# self.f_d2mu_p = nn.ModuleList()
		
		for lev in range(self.n_levels):
			dphi2mu_q_block = nn.ModuleList()
			dphi2sig_q_block = nn.ModuleList()
			last_layer_size = self.d_layers[lev]
			for layer_size in self.posterior_layers:
				dphi2mu_q_block.append(nn.Linear(last_layer_size, layer_size, bias=True))
				dphi2mu_q_block.append(feedforward_actfun_rnn())
				dphi2sig_q_block.append(nn.Linear(last_layer_size, layer_size, bias=True))
				dphi2sig_q_block.append(feedforward_actfun_rnn())
				last_layer_size = layer_size
			dphi2mu_q_block.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
			dphi2sig_q_block.append(nn.Linear(last_layer_size, self.z_layers[lev], bias=True))
			# Softplus version
			dphi2sig_q_block.append(nn.Softplus())
		
			self.f_dphi2mu_q_block.append(nn.Sequential(*dphi2mu_q_block))
			if not isinstance(self.sig_scale, float):
				self.f_dphi2sig_q_block.append(nn.Sequential(*dphi2sig_q_block))
	
				
		###################################################
		# recurrent connections
		if self.rnn_type == 'mtgru':
			##### RNN for model learning
			self.rnn_model_levels = nn.ModuleList()
			self.rnn_model_levels.append( nn.GRUCell(self.top_k*self.d_layers[0], self.d_layers[0]) )
		else:
			raise ValueError("rnn_type must be 'mtgru'")
			
		##### Optimizer
		# self.optimizer_st = torch.optim.Adam(self.parameters(), lr=lr_st)
		
		self.optimizer_st = torch.optim.Adam([*self.trans_model_q.parameters(), *self.f_x2phi.parameters(),
		                                      *self.rnn_model_levels.parameters(),
		                                        *self.f_dphi2mu_q_block.parameters(),
		                                      *self.f_dphi2sig_q_block.parameters(),
		                                      *self.rws_logit.parameters()], lr=lr_st)

		

	def rnn_model(self, prev_d_levels, q_output_selected):
		new_d_levels = []

		if self.rnn_type == 'mtgru':  # gru or lstm
			new_d = self.rnn_model_levels[0](q_output_selected, prev_d_levels[0])
			new_d_levels.append(new_d)
		else:
			raise NotImplementedError

		return new_d_levels
	

	def sample_z(self, mu, sig):
		# Using reparameterization trick to sample from a gaussian
		if isinstance(sig, torch.Tensor):
			eps = Variable(torch.randn_like(mu))
		else:
			eps = torch.randn_like(mu)
		return mu + sig * eps
	
	
	def forward_inference_model(self, prev_d_levels, x_obs, a_prev_obs):
		
		# feature extraction
		x_phi = self.f_x2phi(x_obs)  # (1,256), (4, 2, 256)
		
		if len(x_phi.size()) < 3:
			x_phi = x_phi.unsqueeze(dim=1)  # (1,1,256)
		if len(a_prev_obs.size()) < 3:
			a_prev_obs = a_prev_obs.unsqueeze(dim=1)  # (1,1,2)
		
		# print("x_phi", x_phi.shape)
		# print("a_prev_obs", a_prev_obs.shape)
		# print("check 1", x_phi[:, 0, :].shape)
		# print("check 2", x_phi[:, 0, :].squeeze().shape)
		
		assert len(x_phi.size()) == 3
		assert len(a_prev_obs.size()) == 3
		
		### Concatenate x_phi and action
		x_phi_action = torch.cat((x_phi, a_prev_obs), dim=-1)

		# Step 1. input obs and action to Attention for model_q.
		# Then select some of them. Default is to choose 2 ends like bi-LSTM (block_len >=2).
		# Later, we are going to choose 'self.trans_select_N' number of outputs based on Attention score.
		
		trans_q_output, attention_matrix = self.trans_model_q.forward(x_phi_action.permute(1, 0, 2)) # (seq, batch_size=4, 256)
		trans_q_output = trans_q_output.permute(1, 0, 2) # (batch=4, seq, 256)
		# print("trans_q_output", trans_q_output, trans_q_output.shape) # (batch, seq, hidden_dim=256)
		# print("attention", attention_matrix, attention_matrix.shape) # (batch*n_head, trans_seq, trans_seq)

		attention_matrix_align = torch.cat(attention_matrix.split(x_phi.size()[0], dim=0), 2) # (batch, trans_seq, trans_seq*n_head)
		# print("attention_matrix_align", attention_matrix_align.shape)

		# ### Pass only the largest one
		# _, max_index = torch.max(attention_matrix_align.sum(dim=-1), dim=-1, keepdim=True)
		# # print("top value", max_value, max_value.shape) # (batch, 1)
		# # print("top index", max_index, max_index.shape) # (batch, 1)
		#
		# max_index_repeat = max_index.unsqueeze(dim=-1).repeat(1, 1, self.d_layers[0]) # (batch, 1, 256)
		# q_output_selected = torch.gather(trans_q_output, dim=1, index=max_index_repeat).squeeze(dim=1) # (batch, hidden_dim=256)
		
		
		### Pass top K elements
		batch_here = trans_q_output.shape[0]
		seq_len_here = trans_q_output.shape[1]
	
		# print("trans_q_output", trans_q_output.shape, trans_q_output.shape[1]) # (batch=4, seq, 256)
		_, top_k_index = torch.topk(attention_matrix_align.sum(dim=-1), k=min(self.top_k, seq_len_here), dim=-1)
		# print("top k", top_k_index.shape) # (batch, min(self.top_k, trans_q_output.shape[1]) )
		# print()
		
		top_k_index_repeat = top_k_index.unsqueeze(dim=-1).repeat(1, 1, self.d_layers[0])
		# (batch, min(self.top_k, trans_q_output.shape[1]), 256)
		top_k_q_output_selected = torch.gather(trans_q_output, dim=1, index=top_k_index_repeat) # selected vectors
		# (batch, min(self.top_k, trans_q_output.shape[1]), 256)
		# print("top_k_q_output_selected", top_k_q_output_selected, top_k_q_output_selected.shape)
		
		reshaped = torch.reshape(top_k_q_output_selected, (batch_here, -1)) # (batch, min(self.top_k, trans_q_output.shape[1])*256)
		# print("reshaped init", reshaped, reshaped.shape)
		
		## Padding if necessary
		if seq_len_here < self.top_k:
			# print("Check~~~~~~~~~~~~~~~~~~~~~~~~~~~")
			zero_pad = torch.zeros( batch_here, (self.top_k - seq_len_here) * trans_q_output.shape[2] )
			reshaped = torch.cat((reshaped, zero_pad), dim=-1) # (batch, self.top_k*256)
		
		# print("reshaped", reshaped, reshaped.shape)
		# print()
		
		# Step 2. Here, 'prev_d_levels' should be changed to block_variable recurrently.
		new_d_levels = self.rnn_model(prev_d_levels, reshaped)  # (batch, hidden=256)
		
		return new_d_levels, reshaped
	
	
	def block_mu_sig(self, new_d_levels):
		
		block_mu_levels = [self.f_dphi2mu_q_block[l](new_d_levels[l]) for l in range(self.n_levels)]
		## Softplus version
		block_sig_levels = [self.f_dphi2sig_q_block[l](new_d_levels[l]) for l in range(self.n_levels)]
		
		# log_sigma version
		# block_sig_levels = [torch.exp(self.f_dphi2sig_q_block[l](new_d_levels[l]).clamp(LOG_STD_MIN_st, LOG_STD_MAX_st)) for l in range(self.n_levels)]

		return block_mu_levels, block_sig_levels
	
	
	def train_st(self, x_obs, a_obs, d_levels_0=None, h_0_detach=True, validity=None, done_obs=None,
	             seq_len=64, trans_seq_interval=1):
		
		### shorten x, r .. by using v
		if not validity is None:
			v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
			stps = np.sum(v, axis=1)
			max_stp = int(np.max(stps))
			
			x_obs = x_obs[:, :max_stp]
			a_obs = a_obs[:, :max_stp]
			
			if not done_obs is None:
				done_obs = done_obs[:, :max_stp]
			
			validity = validity[:, :max_stp].reshape([x_obs.size()[0], x_obs.size()[1]])
		
		batch_size = x_obs.size()[0]
		
		if validity is None:  # no need for padding
			validity = torch.ones([x_obs.size()[0], x_obs.size()[1]], requires_grad=False)
		
		# if h_levels_0 is None:
		#     h_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
		# elif isinstance(h_levels_0[0], np.ndarray):
		#         h_levels_0 = [torch.from_numpy(h_0) for h_0 in h_levels_0]
		
		if d_levels_0 is None:
			d_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
		elif isinstance(d_levels_0[0], np.ndarray):
			d_levels_0 = [torch.from_numpy(d_0) for d_0 in d_levels_0]
		
		if h_0_detach:
			# h_levels_init = [h_0.detach() for h_0 in h_levels_0]
			d_levels_init = [d_0.detach() for d_0 in d_levels_0]
			# h_levels = h_levels_init
			d_levels = d_levels_init
		else:
			# h_levels_init = [h_0 for h_0 in h_levels_0]
			d_levels_init = [d_0 for d_0 in d_levels_0]
			# h_levels = h_levels_init
			d_levels = d_levels_init
		
		### z_levels added. Detach false
		z_levels_0 = [torch.zeros((batch_size, z_size)) for z_size in self.z_layers]
		z_levels = [z_0 for z_0 in z_levels_0]
		
		x_obs = x_obs.data
		a_obs = a_obs.data
		if not done_obs is None:
			done_obs = done_obs.data
		
		# sample minibatch of minibatch_size x seq_len
		stps_burnin = 64
		x_sampled = torch.zeros([x_obs.size()[0], seq_len, x_obs.size()[-1]], dtype=torch.float32)
		a_sampled = torch.zeros([a_obs.size()[0], seq_len, a_obs.size()[-1]], dtype=torch.float32)
		# v_sampled = torch.zeros([validity.size()[0], seq_len], dtype=torch.float32)
		
		for b in range(x_obs.size()[0]):
			v = validity.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
			stps = np.sum(v[b], axis=0).astype(int)
			start_index = np.random.randint(-seq_len + 1, stps - 1)
			
			for tmp, TMP in zip((x_sampled, a_sampled), (x_obs, a_obs)):
				
				if start_index < 0 and start_index + seq_len > stps:
					tmp[b, :stps] = TMP[b, :stps]
				
				elif start_index < 0:
					tmp[b, :(start_index + seq_len)] = TMP[b, :(start_index + seq_len)]
				
				elif start_index + seq_len > stps:
					tmp[b, :(stps - start_index)] = TMP[b, start_index:stps]
				
				else:
					tmp[b] = TMP[b, start_index: (start_index + seq_len)]
			
			# h_levels_b = [h_level[b:b+1] for h_level in h_levels]
			d_levels_b = [d_level[b:b + 1] for d_level in d_levels]
			z_levels_b = [z_level[b:b + 1] for z_level in z_levels]
			
			if start_index < 1:
				pass
			else:
				x_tmp = x_obs[b:b + 1,
				        max(0, start_index - stps_burnin):start_index]  # torch.Size([1, 16, 13]) torch.Size([1, 16, 2])
				a_tmp = a_obs[b:b + 1, max(0, start_index - stps_burnin):start_index]
				# print("shape", x_tmp.shape, a_tmp.shape)
				
				for t_burnin in range(0, x_tmp.size()[1], trans_seq_interval):  ## trans_seq_interval should divide x_tmp.size()[1]
					
					curr_x_obs = x_tmp[:, t_burnin:t_burnin + trans_seq_interval]  ## (batch=4, trans_seq_interval=2, dim=13)
					prev_a_obs = a_tmp[:, t_burnin:t_burnin + trans_seq_interval]
					
					d_levels_b, _ = self.forward_inference_model(d_levels_b, curr_x_obs, prev_a_obs)  ### Should be revised
						
				for lev in range(self.n_levels):
					# h_levels[lev][b] = h_levels_b[lev][0].data
					d_levels[lev][b] = d_levels_b[lev][0].data
					z_levels[lev][b] = z_levels_b[lev][0].data
		
		# d_series_levels = [[] for l in range(self.n_levels)]
		# sig_q_series_levels = [[] for l in range(self.n_levels)]
		# mu_q_series_levels = [[] for l in range(self.n_levels)]
		
		### Transformer added
		loss_p = torch.zeros(1)[0]
		loss_q = torch.zeros(1)[0]
		
		for stp in range(0, seq_len, trans_seq_interval):
			
			curr_x_obs = x_sampled[:, stp:stp + trans_seq_interval]  ## (batch=4, trans_seq_interval=2, dim=13), ## (batch=4, trans_seq_interval=1, dim=13) check
			prev_a_obs = a_sampled[:, stp:stp + trans_seq_interval]  ## (batch=4, trans_seq_interval=2, dim=2)
			# curr_v_sampled = v_sampled[:, stp:stp + trans_seq_interval]
			
			# a_prev = prev_a_obs if self.action_feedback else None
			
			if not isinstance(self.sig_scale, float): ### Here!
				# _, _, z_p_levels, mu_p_levels, sig_p_levels, mux_pred, sigx_pred = self.forward_generative(
				#     d_levels, curr_x_obs, a_prev, z_levels)
				d_levels, q_output_select = self.forward_inference_model(d_levels, curr_x_obs, prev_a_obs)  # z_levels is the last previous latent sample
			else:  ### Neglect this part
				# _, _, z_p_levels, mu_p_levels, sig_p_levels, _, _ = self.forward_generative(
				#     d_levels, curr_x_obs, a_prev, z_levels)
				d_levels_new, q_output_select = self.forward_inference_model(d_levels, curr_x_obs, prev_a_obs)
				d_levels = d_levels_new
				
			
			block_q_mu_levels, block_q_sig_levels = self.block_mu_sig(d_levels)
			# print("block mu", block_q_mu_levels[0].shape) # [4,64]
			# print("block sig", block_q_sig_levels[0].shape) # [4,64]
			
			block_q_prob = dis.Normal(block_q_mu_levels[0], block_q_sig_levels[0]) #[4, 64]
			block_variable_samples = block_q_prob.rsample(sample_shape=[self.K]) # [self.K, 4, 64]
			
			logit_q = torch.sum(block_q_prob.log_prob(block_variable_samples), dim=-1).permute(1,0) # [4, self.K]
			# logit_q = torch.sum(block_q_prob.log_prob(block_variable_samples).clamp(-20, 10), dim=-1).permute(1,0) # [4, self.K]
			
			##### original
			q_output_select_tile = q_output_select.detach().unsqueeze(dim=0).repeat(self.K, 1, 1) # [self.K, 4, 256*self.top_k]
			# q_output_select_tile = q_output_select.unsqueeze(dim=0).repeat(self.K, 1, 1) # [self.K, 4, 256*self.top_k]
			logit_p = self.rws_logit(torch.cat((q_output_select_tile, block_variable_samples.detach()), dim=-1)).squeeze(dim=-1).permute(1,0) # [4, self.K]
			

			### We now have logit_p tensor w.r.t. theta, and logit_q tensor w.r.t. phi.

			### !!! Important! Weight should be sampled 'values', so we use .detach().
			log_weight = (logit_p - logit_q).detach()

			log_weight_max, log_weight_amax = torch.max(log_weight, dim=-1, keepdim=True)  # [batch, 1]
			### leave log_weight_amax for later ablation study.
			
			# print("log_weight - log_weight_max", log_weight - log_weight_max) # [batch, self.K] by broadcasting
			# print("sum", torch.sum( torch.exp( log_weight - log_weight_max ), dim=-1, keepdim=True  )) # [batch, 1]
			
			log_sum_weight = log_weight_max + torch.log(
				torch.sum(torch.exp(log_weight - log_weight_max), dim=-1, keepdim=True))  # [batch, 1]
			# print("log_sum_weight", log_sum_weight, log_sum_weight.shape)
			
			self_normalized_weight = torch.exp(log_weight - log_sum_weight)  # [batch, self.K] by broadcasting
			# print("self_normalized_weight", self_normalized_weight, torch.sum(self_normalized_weight, dim=-1)) # sum to 1
			# print("~~~~~~~~~")
			# print()
			
			### detach check
			# self_normalized_weight = self_normalized_weight.detach()
			
			##### Step 2. Calculate 'loss' log p_theta and log q_phi.
			
			# print("logit_p", logit_p, logit_p.shape) # [batch=4, sample_num_K]
			# print("logit_q", logit_q, logit_q.shape)

			weighted_logit_p = self_normalized_weight * logit_p
			weighted_logit_q = self_normalized_weight * logit_q
			
			# print("inner size", torch.mean(torch.sum(weighted_logit_p, dim=-1)), torch.mean(torch.sum(weighted_logit_p, dim=-1)).shape)
			
			loss_p += torch.mean(torch.sum(weighted_logit_p, dim=-1))
			loss_q += torch.mean(torch.sum(weighted_logit_q, dim=-1))
		
		## reverse sign to minimize
		loss_total = -(loss_p + loss_q)

		self.optimizer_st.zero_grad()
		loss_total.backward()
		self.optimizer_st.step()

		return [loss_p.cpu().item(), loss_q.cpu().item()], d_levels_init
	
	def init_hidden_zeros(self, batch_size=1):
		
		h_levels = [torch.zeros((batch_size, d_size)) for d_size in self.d_layers]
		
		return h_levels


class RL_agent(nn.Module):
	def __init__(self,
	             klm: Block_model,
	             lr_rl=3e-4,
	             gamma=0.99,
	             feedforward_actfun_sac=nn.ReLU,
	             beta_h='auto_1.0',
	             policy_layers=[256, 256],
	             value_layers=[256, 256]):

		super(RL_agent, self).__init__()
		
		self.gamma = gamma
		self.a_prev = None
		
		self.klm = klm
		self.action_size = self.klm.action_size
		self.input_size = self.klm.input_size
		
		self.include_obs = True
		self.policy_layers = policy_layers
		self.value_layers = value_layers
		
		self.d_layers = []
		for lev in range(self.klm.n_levels):
			self.d_layers.append( self.klm.d_layers[lev] )
		
		self.n_levels = len(self.d_layers)
		
		self.forward_inference_klm_model = self.klm.forward_inference_model
		
		# self.h_levels = self.init_hidden_zeros(batch_size=1)  # [1, 256]
		self.d_levels = self.init_hidden_zeros(batch_size=1)  # [1, 256]
		self.z_levels = [torch.zeros((1, z_size_klm)) for z_size_klm in self.klm.z_layers]  # [1, 64]

		self.block_mu_sig = self.klm.block_mu_sig
		
		self.f_x2phi = self.klm.f_x2phi
		
		
		
		### RL element generation
		
		###################################################
		### Feature extraction
		
		# feature-extracting transformations
		x2phi_rl = nn.ModuleList()
		last_layer_size = self.input_size
		for layer_size in self.d_layers:
			x2phi_rl.append(nn.Linear(last_layer_size, layer_size, bias=True))
			last_layer_size = layer_size
			x2phi_rl.append(nn.Tanh())
		x2phi_rl.append(nn.Linear(last_layer_size, self.d_layers[0] - self.action_size, bias=True))  ### last layer: 128->126

		self.f_x2phi_rl = nn.Sequential(*x2phi_rl)
		
		self.rnn_levels_rl = nn.GRU(input_size= self.d_layers[0] + 2 * self.klm.z_layers[0],
		                            hidden_size= self.d_layers[0]) # input 13 + 2 + 2*
		
		self.beta_h = beta_h
		self.target_entropy = - np.float32(self.action_size)
		
		if isinstance(self.beta_h, str) and self.beta_h.startswith('auto'):
			# Default initial value of beta_h when learned
			init_value = 1.0
			if '_' in self.beta_h:
				init_value = float(self.beta_h.split('_')[1])
				assert init_value > 0., "The initial value of beta_h must be greater than 0"
			
			self.log_beta_h = torch.tensor(np.log(init_value).astype(np.float32), requires_grad=True)
			# self.beta_h = torch.exp(self.log_beta_h)
		
		# policy network
		self.d2mua = nn.ModuleList()
		last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
		for layer_size in self.policy_layers:
			self.d2mua.append(nn.Linear(last_layer_size, layer_size, bias=True))
			last_layer_size = layer_size
			self.d2mua.append(feedforward_actfun_sac())
		self.d2mua.append(nn.Linear(last_layer_size, self.action_size, bias=True))
		# self.d2mua.append(nn.Tanh())
		
		self.f_d2mua = nn.Sequential(*self.d2mua)
		
		self.d2log_siga = nn.ModuleList()
		last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
		for layer_size in self.policy_layers:
			self.d2log_siga.append(nn.Linear(last_layer_size, layer_size, bias=True))
			last_layer_size = layer_size
			self.d2log_siga.append(feedforward_actfun_sac())
		self.d2log_siga.append(nn.Linear(last_layer_size, self.action_size, bias=True))
		
		self.f_d2log_siga = nn.Sequential(*self.d2log_siga)
		
		# V network
		self.d2v = nn.ModuleList()
		last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
		for layer_size in self.value_layers:
			self.d2v.append(nn.Linear(last_layer_size, layer_size, bias=True))
			last_layer_size = layer_size
			self.d2v.append(feedforward_actfun_sac())
		self.d2v.append(nn.Linear(last_layer_size, 1, bias=True))
		
		self.f_d2v = nn.Sequential(*self.d2v)
		
		# Q networks (double q-learning)
		self.da2q1 = nn.ModuleList()
		last_layer_size = self.d_layers[0] + self.action_size if not self.include_obs else self.d_layers[
			                                                                                   0] + self.input_size + self.action_size
		for layer_size in self.value_layers:
			self.da2q1.append(nn.Linear(last_layer_size, layer_size, bias=True))
			last_layer_size = layer_size
			self.da2q1.append(feedforward_actfun_sac())
		self.da2q1.append(nn.Linear(last_layer_size, 1, bias=True))
		
		self.f_da2q1 = nn.Sequential(*self.da2q1)
		
		self.da2q2 = nn.ModuleList()
		last_layer_size = self.d_layers[0] + self.action_size if not self.include_obs else self.d_layers[
			                                                                                   0] + self.input_size + self.action_size
		for layer_size in self.value_layers:
			self.da2q2.append(nn.Linear(last_layer_size, layer_size, bias=True))
			last_layer_size = layer_size
			self.da2q2.append(feedforward_actfun_sac())
		self.da2q2.append(nn.Linear(last_layer_size, 1, bias=True))
		
		self.f_da2q2 = nn.Sequential(*self.da2q2)
		
		# target V network
		self.d2v_tar = nn.ModuleList()
		last_layer_size = self.d_layers[0] if not self.include_obs else self.d_layers[0] + self.input_size
		for layer_size in self.value_layers:
			self.d2v_tar.append(nn.Linear(last_layer_size, layer_size, bias=True))
			last_layer_size = layer_size
			self.d2v_tar.append(feedforward_actfun_sac())
		self.d2v_tar.append(nn.Linear(last_layer_size, 1, bias=True))
		
		self.f_d2v_tar = nn.Sequential(*self.d2v_tar)
		
		# synchronizing target V network and V network
		state_dict_tar = self.f_d2v_tar.state_dict()
		state_dict = self.f_d2v.state_dict()
		for key in list(self.f_d2v.state_dict().keys()):
			state_dict_tar[key] = state_dict[key]
		self.f_d2v_tar.load_state_dict(state_dict_tar)
		
		# p = prior (generative model), q = posterier (inference model)
		
		self.common_param = [*self.f_x2phi_rl.parameters(), *self.rnn_levels_rl.parameters()]
		
		self.optimizer_a = torch.optim.Adam(self.common_param + [*self.f_d2mua.parameters(), *self.f_d2log_siga.parameters()], lr=lr_rl)
		self.optimizer_v = torch.optim.Adam(
			self.common_param + [*self.f_da2q1.parameters(), *self.f_da2q2.parameters(), *self.f_d2v.parameters()], lr=lr_rl)
		self.optimizer_e = torch.optim.Adam(self.common_param + [self.log_beta_h], lr=lr_rl)  # optimizer for beta_h

		self.mse_loss = nn.MSELoss()
		
		# ## RL agent parameters
		# print()
		# print("self.parameters RL", self.parameters)
		# for item in self.parameters():
		# 	print("learn st size", item.shape)
		# print("~~~~~~~~~~~~~~~~~~~~~~~~")
		
		
	
	def rnn_rl(self, prev_z_levels_rl, prev_block_mu, prev_block_sig, x_obs, a_prev_obs):

		x_phi = self.f_x2phi(x_obs)  # (1,126), (batch, seq, 126)
		
		if len(x_phi.size()) < 3: # Add sequence axis
			x_phi = x_phi.unsqueeze(dim=1)  # (1,1,126)
		if len(a_prev_obs.size()) < 3: # Add sequence axis
			a_prev_obs = a_prev_obs.unsqueeze(dim=1)  # (1,1,2)
		
		assert len(x_phi.size()) == 3
		assert len(a_prev_obs.size()) == 3
		# print("check", prev_z_q[0].shape)

		### Concatenate x_phi and action
		x_phi_action_rl = torch.cat((x_phi, a_prev_obs), dim=-1) # (batch, seq, 128)
		
		seq_len = x_phi_action_rl.size()[1]
		
		prev_block_mu_tile = prev_block_mu[0].detach().unsqueeze(dim=1).repeat(1, seq_len, 1)
		prev_block_sig_tile = prev_block_sig[0].detach().unsqueeze(dim=1).repeat(1, seq_len, 1)

		## current_x_phi_first [4, 256]; prev_z_last [4, 64]
		last = torch.cat((x_phi_action_rl, prev_block_mu_tile, prev_block_sig_tile), dim=-1) # (batch, seq, 256)
		new_outputs, new_z_levels_rl = self.rnn_levels_rl(last.permute(1, 0, 2), prev_z_levels_rl[0].unsqueeze(dim=0))
		
		# new_outputs: (seq, batch, 256)
		# new_z_levels_rl: (1, batch, 256)
		
		return [new_z_levels_rl.squeeze(dim=0)], new_outputs
	
	###
	
	
	def sample_z(self, mu, sig):
		# Using reparameterization trick to sample from a gaussian
		eps = Variable(torch.randn_like(mu))
		return mu + sig * eps
	
	def sample_action(self, d0_prev, x_prev, detach=False):
		# output action
		if not self.include_obs:
			s = d0_prev
		else:
			s = torch.cat((d0_prev, x_prev), dim=-1)
		
		mua = self.f_d2mua(s)
		
		siga = torch.exp(self.f_d2log_siga(s).clamp(LOG_STD_MIN, LOG_STD_MAX))
		
		if detach:
			return torch.tanh(self.sample_z(mua, siga).detach()), mua.detach(), siga.detach()
		else:
			return torch.tanh(self.sample_z(mua, siga).detach()), mua, siga
	
	def preprocess_sac(self, x_obs, r_obs, a_obs, d_obs=None, v_obs=None, seq_len=64, trans_seq_interval=1):
		
		### shorten x, r .. by using v
		if not v_obs is None:
			v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
			stps = np.sum(v, axis=1)
			max_stp = int(np.max(stps))
			
			x_obs = x_obs[:, :max_stp]
			a_obs = a_obs[:, :max_stp]
			r_obs = r_obs[:, :max_stp]
			d_obs = d_obs[:, :max_stp]
			v_obs = v_obs[:, :max_stp]
		
		batch_size = x_obs.size()[0]
		start_indices = np.zeros(x_obs.size()[0], dtype=int)
		
		for b in range(x_obs.size()[0]):
			v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
			stps = np.sum(v[b], axis=0).astype(int)
			start_indices[b] = np.random.randint(-seq_len + 1, stps - 1)
		
		x_obs = x_obs.data  # (4, 128, 13)
		a_obs = a_obs.data
		r_obs = r_obs.data
		d_obs = d_obs.data
		v_obs = v_obs.data
		
		# initialize hidden states
		d_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
		z_levels_0 = self.init_hidden_zeros(batch_size=batch_size)
		# z_levels_0 = [torch.zeros((batch_size, z_size_klm)) for z_size_klm in self.klm.z_layers]
		
		d_levels = [d_0.detach() for d_0 in d_levels_0]
		z_levels = [z_0.detach() for z_0 in z_levels_0]
		

		d_levels_klm = []
		z_levels_klm = []
		
		for lev in range(self.n_levels):
		
			d_levels_klm.append(d_levels[lev])
			z_levels_klm.append(z_levels[lev])
			
		x_sampled = torch.zeros([x_obs.size()[0], seq_len + 1, x_obs.size()[-1]],
		                        dtype=torch.float32)  # +1 for SP # (4, 128, 13)
		a_sampled = torch.zeros([a_obs.size()[0], seq_len + 1, a_obs.size()[-1]], dtype=torch.float32)  # (4, 128, 2)
		

		d_series_levels_klm = [[] for l in range(self.n_levels)]
		
		stps_burnin = 64
		
		for b in range(x_obs.size()[0]):
			v = v_obs.cpu().numpy().reshape([x_obs.size()[0], x_obs.size()[1]])
			stps = np.sum(v[b], axis=0).astype(int)
			start_index = start_indices[b]
			
			for tmp, TMP in zip((x_sampled, a_sampled), (x_obs, a_obs)):

				if start_index < 0 and start_index + seq_len + 1 > stps:
					tmp[b, :stps] = TMP[b, :stps]

				elif start_index < 0:
					tmp[b, :(start_index + seq_len + 1)] = TMP[b, :(start_index + seq_len + 1)]

				elif start_index + seq_len + 1 > stps:
					tmp[b, :(stps - start_index)] = TMP[b, start_index:stps]

				else:
					tmp[b] = TMP[b, start_index: (start_index + seq_len + 1)]
			
			# h_levels_b_klm = [h_level[b:b + 1] for h_level in h_levels_klm]
			d_levels_b_klm = [d_level[b:b + 1] for d_level in d_levels_klm]
			z_levels_b_klm = [z_level[b:b + 1] for z_level in z_levels_klm]
			
			if start_index < 1:
				pass
			else:
				x_tmp = x_obs[b:b + 1, max(0, start_index - stps_burnin):start_index]
				a_tmp = a_obs[b:b + 1, max(0, start_index - stps_burnin):start_index]
				
				for t_burnin in range(x_tmp.size()[0]):
					x_tmp_t = x_tmp[:, t_burnin]
					a_tmp_t = a_tmp[:, t_burnin]
					# a_tmp_t = a_tmp[:, t_burnin] if self.klm.action_feedback else None
					
					### RNN
					block_q_mu_levels_b_klm, block_q_sig_levels_b_klm = self.block_mu_sig(d_levels_b_klm)
					z_levels_b_klm, _ = self.rnn_rl(z_levels_b_klm, block_q_mu_levels_b_klm, block_q_sig_levels_b_klm, x_tmp_t, a_tmp_t)
					
					### Block
					d_levels_b_klm, _ = self.forward_inference_klm_model(d_levels_b_klm, x_tmp_t, a_tmp_t)
					
				for lev in range(self.n_levels):
					# h_levels_klm[lev][b] = h_levels_b_klm[lev][0].data
					d_levels_klm[lev][b] = d_levels_b_klm[lev][0].data
					z_levels_klm[lev][b] = z_levels_b_klm[lev][0].data
		
		#################
		
		for stp in range(0, seq_len + 1, trans_seq_interval):
			curr_x_obs = x_sampled[:, stp:min(seq_len + 1, stp + trans_seq_interval)]  ## (batch=4, trans_seq_interval=2, dim=13)
			prev_a_obs = a_sampled[:, stp:min(seq_len + 1, stp + trans_seq_interval)]  ## (4, 2, 2)
			
			block_q_mu_levels_klm, block_q_sig_levels_klm = self.block_mu_sig(d_levels_klm)
			
			z_levels_klm, output_total \
				= self.rnn_rl(z_levels_klm, block_q_mu_levels_klm, block_q_sig_levels_klm, curr_x_obs, prev_a_obs)
			
			# ## detach() to segregate RL and model learning inside rnn_rl.
			# output_total: (seq, batch, 256)

			d_series_levels_klm[0].append(output_total.permute(1,0,2))

			### Generate next block variable
			d_levels_klm, _ = self.forward_inference_klm_model(d_levels_klm, curr_x_obs, prev_a_obs)
			
		d_low_tensor_klm = torch.cat(d_series_levels_klm[0], dim=1)
		
		S_sampled_klm = d_low_tensor_klm[:, :-1, :]
		SP_sampled_klm = d_low_tensor_klm[:, 1:, :]
		
		if self.include_obs:
			S_sampled = torch.cat((S_sampled_klm, x_sampled[:, :-1, :]), dim=-1)
			SP_sampled = torch.cat((SP_sampled_klm, x_sampled[:, 1:, :]), dim=-1)
		else:
			S_sampled = S_sampled_klm
			SP_sampled = SP_sampled_klm
		
		A = a_obs
		R = r_obs
		
		if d_obs is None:
			D = torch.zeros_like(R, dtype=torch.float32)
		else:
			D = d_obs
		
		if v_obs is None:  # no need for padding
			V = torch.ones_like(R, requires_grad=False, dtype=torch.float32)
		else:
			V = v_obs
		
		A_sampled = torch.zeros([A.size()[0], seq_len + 1, A.size()[-1]], dtype=torch.float32)
		D_sampled = torch.zeros([D.size()[0], seq_len + 1, 1], dtype=torch.float32)
		R_sampled = torch.zeros([R.size()[0], seq_len + 1, 1], dtype=torch.float32)
		V_sampled = torch.zeros([V.size()[0], seq_len + 1, 1], dtype=torch.float32)
		
		for b in range(A.size()[0]):
			v = v_obs.cpu().numpy().reshape([A.size()[0], A.size()[1]])
			stps = np.sum(v[b], axis=0).astype(int)
			start_index = start_indices[b]
			
			for tmp, TMP in zip((A_sampled, D_sampled, R_sampled, V_sampled),
			                    (A, D, R, V)):
				
				if start_index < 0 and start_index + seq_len + 1 > stps:
					tmp[b, :stps] = TMP[b, :stps]
				
				elif start_index < 0:
					tmp[b, :(start_index + seq_len + 1)] = TMP[b, :(start_index + seq_len + 1)]
				
				elif start_index + seq_len + 1 > stps:
					tmp[b, :(stps - start_index)] = TMP[b, start_index:stps]
				
				else:
					tmp[b] = TMP[b, start_index: (start_index + seq_len + 1)]
		
		R_sampled = R_sampled[:, 1:, :].data
		A_sampled = A_sampled[:, 1:, :].data
		D_sampled = D_sampled[:, 1:, :].data
		V_sampled = V_sampled[:, 1:, :].data
		
		return S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled, V_sampled
	
	def train_rl_sac_(self, S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled, V_sampled,
	                  reward_scale=1.0, computation='explicit', grad_clip=False):
		
		gamma = self.gamma
		
		if isinstance(self.beta_h, str):
			beta_h = torch.exp(self.log_beta_h).data
		else:
			beta_h = self.beta_h
		
		mua_tensor = self.f_d2mua(S_sampled)
		siga_tensor = torch.exp(self.f_d2log_siga(S_sampled).clamp(LOG_STD_MIN, LOG_STD_MAX))
		v_tensor = self.f_d2v(S_sampled)
		vp_tensor = self.f_d2v_tar(SP_sampled)
		q_tensor_1 = self.f_da2q1(torch.cat((S_sampled, A_sampled), dim=-1))
		q_tensor_2 = self.f_da2q2(torch.cat((S_sampled, A_sampled), dim=-1))
		
		# ------ explicit computing---------------
		if computation == 'explicit':
			# ------ loss_v ---------------
			
			sampled_u = self.sample_z(mua_tensor.data, siga_tensor.data).data
			sampled_a = torch.tanh(sampled_u)
			
			sampled_q = torch.min(self.f_da2q1(torch.cat((S_sampled, sampled_a), dim=-1)).data,
			                      self.f_da2q2(torch.cat((S_sampled, sampled_a), dim=-1)).data)
			
			q_exp = sampled_q
			log_pi_exp = torch.sum(- (mua_tensor.data - sampled_u.data).pow(2)
			                       / (siga_tensor.data.pow(2)) / 2
			                       - torch.log(siga_tensor.data * torch.tensor(2.5066)),
			                       dim=-1, keepdim=True) # [4,64,1].
						
			log_pi_exp -= torch.sum(torch.log(1.0 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
			
			v_tar = (q_exp - beta_h * log_pi_exp.data).detach().data
			
			loss_v = 0.5 * self.mse_loss(v_tensor * V_sampled, v_tar * V_sampled)
			
			loss_q = 0.5 * self.mse_loss(q_tensor_1 * V_sampled, (
			reward_scale * R_sampled + (1 - D_sampled) * gamma * vp_tensor.detach().data) * V_sampled) \
			         + 0.5 * self.mse_loss(q_tensor_2 * V_sampled, (
			reward_scale * R_sampled + (1 - D_sampled) * gamma * vp_tensor.detach().data) * V_sampled)
			
			loss_critic = loss_v + loss_q
			
			# -------- loss_a ---------------
			
			sampled_u = Variable(self.sample_z(mua_tensor.data, siga_tensor.data), requires_grad=True)
			sampled_a = torch.tanh(sampled_u)
			
			Q_tmp = torch.min(self.f_da2q1(torch.cat((S_sampled, torch.tanh(sampled_u)), dim=-1)),
			                  self.f_da2q2(torch.cat((S_sampled, torch.tanh(sampled_u)), dim=-1)))
			Q_tmp.backward(torch.ones_like(Q_tmp), retain_graph=True)
			
			PQPU = sampled_u.grad  # \frac{\partial Q}{\partial a}
			
			eps = (sampled_u.data - mua_tensor.data) / (siga_tensor.data)  # action noise quantity
			a = sampled_a.data  # action quantity
			
			grad_mua = (beta_h * (2 * a) - PQPU).data * V_sampled.repeat_interleave(a.size()[-1], dim=-1) \
			           + REG * mua_tensor * V_sampled.repeat_interleave(a.size()[-1], dim=-1)
			grad_siga = (- (beta_h / (siga_tensor.data + EPS)) + 2 * beta_h * a * eps - PQPU * eps).data \
			            * V_sampled.repeat_interleave(a.size()[-1],
			                                          dim=-1) + REG * siga_tensor * V_sampled.repeat_interleave(
				a.size()[-1], dim=-1)
			
			self.optimizer_v.zero_grad()
			loss_critic.backward(retain_graph=True)
			if grad_clip:
				nn.utils.clip_grad_norm_(
					self.common_param + [*self.f_d2v.parameters(), *self.f_da2q1.parameters(), *self.f_da2q2.parameters()], 1.0)
			self.optimizer_v.step()
			
			self.optimizer_a.zero_grad()
			mua_tensor.backward(grad_mua / torch.ones_like(mua_tensor, dtype=torch.float32).sum(), retain_graph=True)
			siga_tensor.backward(grad_siga / torch.ones_like(siga_tensor, dtype=torch.float32).sum())
			if grad_clip:
				nn.utils.clip_grad_value_(self.common_param + [*self.f_d2log_siga.parameters(), *self.f_d2mua.parameters()], 1.0)
			self.optimizer_a.step()
		
		
		# Using Torch API for computing log_prob
		# ------ implicit computing---------------
		
		elif computation == 'implicit':
			# --------- loss_v ------------
			mu_prob = dis.Normal(mua_tensor, siga_tensor)
			
			sampled_u = mu_prob.sample()
			sampled_a = torch.tanh(sampled_u)
			
			log_pi_exp = torch.sum(mu_prob.log_prob(sampled_u), dim=-1, keepdim=True) - torch.sum(
				torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
			
			sampled_q = torch.min(self.f_da2q1(torch.cat((S_sampled, sampled_a), dim=-1)).data,
			                      self.f_da2q2(torch.cat((S_sampled, sampled_a), dim=-1)).data)
			q_exp = sampled_q
			
			v_tar = (q_exp - beta_h * log_pi_exp.data).detach().data
			
			loss_v = 0.5 * self.mse_loss(v_tensor * V_sampled, v_tar * V_sampled)
			
			loss_q = 0.5 * self.mse_loss(q_tensor_1 * V_sampled, (
			reward_scale * R_sampled + (1 - D_sampled) * gamma * vp_tensor.detach().data) * V_sampled) \
			         + 0.5 * self.mse_loss(q_tensor_2 * V_sampled, (
			reward_scale * R_sampled + (1 - D_sampled) * gamma * vp_tensor.detach().data) * V_sampled)
			
			loss_critic = loss_v + loss_q
			
			# ----------- loss_a ---------------
			mu_prob = dis.Normal(mua_tensor, siga_tensor)
			
			sampled_u = mu_prob.rsample()
			sampled_a = torch.tanh(sampled_u)
			
			log_pi_exp = torch.sum(mu_prob.log_prob(sampled_u).clamp(-20, 10), dim=-1, keepdim=True) - torch.sum(
				torch.log(1 - sampled_a.pow(2) + EPS), dim=-1, keepdim=True)
			
			loss_a = torch.mean(beta_h * log_pi_exp * V_sampled -
			                    torch.min(self.f_da2q1(torch.cat((S_sampled, sampled_a), dim=-1)),
			                              self.f_da2q2(torch.cat((S_sampled, sampled_a), dim=-1))) * V_sampled) \
			         + REG / 2 * (
			torch.mean((siga_tensor * V_sampled.repeat_interleave(siga_tensor.size()[-1], dim=-1)).pow(2))
			+ torch.mean((mua_tensor * V_sampled.repeat_interleave(mua_tensor.size()[-1], dim=-1)).pow(2)))
			
			self.optimizer_v.zero_grad()
			loss_critic.backward()
			if grad_clip:
				nn.utils.clip_grad_norm_(
					[*self.f_d2v.parameters(), *self.f_da2q1.parameters(), *self.f_da2q2.parameters()], 1.0)
			self.optimizer_v.step()
			
			self.optimizer_a.zero_grad()
			loss_a.backward()
			if grad_clip:
				nn.utils.clip_grad_value_([*self.f_d2log_siga.parameters(), *self.f_d2mua.parameters()], 1.0)
			self.optimizer_a.step()
		# --------------------------------------------------------------------------
		
		# update entropy coefficient if required
		if isinstance(beta_h, torch.Tensor):
			self.optimizer_e.zero_grad()
			
			loss_e = - torch.mean(self.log_beta_h * (log_pi_exp + self.target_entropy).data)
			loss_e.backward()
			self.optimizer_e.step()
		
		# update target V network
		state_dict_tar = self.f_d2v_tar.state_dict()
		state_dict = self.f_d2v.state_dict()
		for key in list(self.f_d2v.state_dict().keys()):
			state_dict_tar[key] = 0.995 * state_dict_tar[key] + 0.005 * state_dict[key]
			# state_dict_tar[key] = 0 * state_dict_tar[key] + 1 * state_dict[key]
		# self.f_d2v_tar.load_state_dict(state_dict)
		self.f_d2v_tar.load_state_dict(state_dict_tar)
		
		if computation == 'implicit':
			return loss_v.item(), loss_a.item(), loss_q.item()
		elif computation == 'explicit':
			return loss_v.item(), torch.mean(grad_mua).item() + torch.mean(grad_siga).item(), loss_q.item()
	
	def init_hidden_zeros(self, batch_size=1):
		
		h_levels = [torch.zeros((batch_size, d_size)) for d_size in self.d_layers]
		
		return h_levels
	
	def detach_states(self, states):
		states = [s.detach() for s in states]
		return states
	
	def init_episode(self, x_0=None, h_levels_0=None, d_levels_0=None):
		# if h_levels_0 is None:
		# 	self.h_levels = self.init_hidden_zeros(batch_size=1)
		# else:
		# 	self.h_levels = [torch.from_numpy(h0) for h0 in h_levels_0]
		
		### Block variable
		if d_levels_0 is None:
			self.d_levels = self.init_hidden_zeros(batch_size=1) # [1,512]
		else:
			self.d_levels = [torch.from_numpy(d0) for d0 in d_levels_0]
		
		### RL state variable
		self.z_levels = self.init_hidden_zeros(batch_size=1)  # [1,512]
		# self.z_levels = [torch.zeros((1, z_size_klm)) for z_size_klm in self.klm.z_layers]
		
		if x_0 is None:
			x_obs_0 = None
		else:
			x_obs_0 = torch.from_numpy(x_0).view(1, -1)
		
		a, _, _ = self.sample_action(self.z_levels[0], x_obs_0, detach=True)
		
		self.a_prev = a
		
		return a.cpu().numpy()
	
	
	
	def select(self, s, r_prev, s_block=None, r_prev_block=None, action_prev_block=None, action_return='normal'):
		### Here, batch_size = 1.
		r_prev = np.array([r_prev]).reshape([-1]).astype(np.float32)
		s = np.array(s).reshape([-1]).astype(np.float32)
		x_obs = torch.cat((torch.from_numpy(s), torch.from_numpy(r_prev))).view([1, -1])

		self.block_mu_select, self.block_sig_select = self.block_mu_sig(self.d_levels)
		self.z_levels, _ = self.rnn_rl(self.z_levels, self.block_mu_select, self.block_sig_select, x_obs, self.a_prev)
		
		a, mua, siga = self.sample_action(self.z_levels[0], x_obs, detach=True)
		
		self.a_prev = a
		
		
		if s_block is not None: #curr_seq_interval % trans_seq_interval == 0:
			
			r_prev_block = np.array([r_prev_block]).reshape([-1, 1]).astype(np.float32)  # (l, 1), 1 <= l <= L
			# print("r_prev_block", r_prev_block.shape)
			s_block = np.array(s_block).astype(np.float32)  # (l, dim=12), 1 <= l <= L
			# print("s", s.shape)
			x_obs_block = torch.cat((torch.from_numpy(s_block), torch.from_numpy(r_prev_block)), dim=1).unsqueeze(dim=0)  # [1, l, 13]
			# print("x_obs", x_obs.shape)
			
			action_prev_block = torch.from_numpy(np.array(action_prev_block).astype(np.float32)).unsqueeze(dim=0)  # [1, l, 2]
			# print("action_prev", action_prev.shape)
			
			### New block variable for next block is generated.
			self.d_levels, _ = self.forward_inference_klm_model(self.d_levels, x_obs_block, action_prev_block)
			
		if action_return == 'mean':
			return torch.tanh(mua).cpu().numpy()
		else:
			return a.cpu().numpy()
	

	def learn_st(self, SP, A, R, D=None, V=None,
	             H0=None, D0=None, times=1, minibatch_size=4, seq_len=64,
	             trans_seq_interval=1):  # learning from the data of this episode
		
		if D is None:
			D = np.zeros_like(R, dtype=np.float32)
		if V is None:
			V = np.ones_like(R, dtype=np.float32)
		
		for xt in range(times):
			weights = np.sum(V, axis=-1) + 2 * seq_len - 2
			e_samples = np.random.choice(SP.shape[0], minibatch_size, p=weights / weights.sum())
			
			sp = SP[e_samples]
			a = A[e_samples]
			r = R[e_samples]
			d = D[e_samples]
			v = V[e_samples]
			
			# if not H0 is None:
			#     h0 = [hl[e_samples] for hl in H0]
			# else:
			#     h0 = None
			
			if not D0 is None:
				d0 = [dl[e_samples] for dl in D0]
			else:
				d0 = None
				
			r_obs = torch.from_numpy(r.reshape([r.shape[0], r.shape[1], 1]))
			x_obs = torch.cat((torch.from_numpy(sp), r_obs), dim=-1)
			
			a_obs = torch.from_numpy(a)
			
			d_obs = torch.from_numpy(d.reshape([r.shape[0], r.shape[1], 1]))
			
			v_obs = torch.from_numpy(v.reshape([r.shape[0], r.shape[1], 1]))
			
			loss, d_levels_init = self.klm.train_st(x_obs, a_obs, validity=v_obs,
			                                        d_levels_0=d0, h_0_detach=False, done_obs=d_obs,
			                                        seq_len=seq_len, trans_seq_interval=trans_seq_interval)

		if not H0 is None:
			for l in range(len(H0)):
				# H0[l][e_samples, :] = h_levels_init[l].cpu().detach().numpy()
				D0[l][e_samples, :] = d_levels_init[l].cpu().detach().numpy()
		
		return H0, D0, loss
	
	def learn_rl_sac(self, SP, A, R, D=None, V=None, H0=None, D0=None, times=1, minibatch_size=4, seq_len=64,
	                 trans_seq_interval=1,
	                 reward_scale=1.0, computation='explicit', grad_clip=False):
		
		if D is None:
			D = np.zeros_like(R, dtype=np.float32)
		if V is None:
			V = np.ones_like(R, dtype=np.float32)
		
		for xt in range(times):
			
			weights = np.sum(V, axis=-1) + 2 * seq_len - 2
			e_samples = np.random.choice(SP.shape[0], minibatch_size, p=weights / weights.sum())
			
			sp = SP[e_samples]
			a = A[e_samples]
			r = R[e_samples]
			d = D[e_samples]
			v = V[e_samples]
			
			if not H0 is None:
				h0 = [hl[e_samples] for hl in H0]
			else:
				h0 = None
			
			if not D0 is None:
				d0 = [dl[e_samples] for dl in D0]
			else:
				d0 = None
			
			r_obs = torch.from_numpy(r.reshape([r.shape[0], r.shape[1], 1]))
			x_obs = torch.cat((torch.from_numpy(sp), r_obs), dim=-1)  # [4, 128, 13]
			
			a_obs = torch.from_numpy(a)
			
			d_obs = torch.from_numpy(d.reshape([r.shape[0], r.shape[1], 1]))
			v_obs = torch.from_numpy(v.reshape([r.shape[0], r.shape[1], 1]))
			
			S_sampled, SP_sampled, A_sampled, R_sampled, D_sampled, V_sampled \
				= self.preprocess_sac(x_obs, r_obs, a_obs, d_obs=d_obs, v_obs=v_obs, seq_len=seq_len,
				                      trans_seq_interval=trans_seq_interval)
			
			loss_v, loss_a, loss_q = self.train_rl_sac_(S_sampled=S_sampled,
			                                            SP_sampled=SP_sampled,
			                                            A_sampled=A_sampled,
			                                            R_sampled=R_sampled,
			                                            D_sampled=D_sampled,
			                                            V_sampled=V_sampled,
			                                            computation=computation,
			                                            reward_scale=reward_scale,
			                                            grad_clip=grad_clip)
		
		return loss_v, loss_a, loss_q
