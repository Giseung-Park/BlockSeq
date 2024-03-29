import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

### Add block model components
from attention_model import BaseModel

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
	classname = m.__class__.__name__
	if classname.find("Linear") != -1:
		m.weight.data.normal_(0, 1)
		m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
		if m.bias is not None:
			m.bias.data.fill_(0)


class ACModel_Block(nn.Module, torch_ac.RecurrentACModel_Block):
	def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
		super().__init__()
		
		# Decide which components are enabled
		self.use_text = use_text
		self.use_memory = use_memory
		
		# Define image embedding
		self.image_conv = nn.Sequential(
			nn.Conv2d(3, 16, (2, 2)),
			nn.ReLU(),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(16, 32, (2, 2)),
			nn.ReLU(),
			nn.Conv2d(32, 64, (2, 2)),
			nn.ReLU()
		)
		n = obs_space["image"][0]
		m = obs_space["image"][1]
		self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
		
		### Block latent variable(embedding) dimension
		self.block_mu_size = 64
		# self.block_hid_dim = 256  # or 512
		
		# Define memory
		if self.use_memory:
			## Stepwise RNN
			self.memory_rnn = nn.LSTMCell(self.image_embedding_size + self.block_mu_size*2, self.semi_memory_size)
			# self.memory_rnn = nn.LSTMCell(self.image_embedding_size + self.block_hid_dim, self.semi_memory_size)
		
		# Define text embedding
		if self.use_text:
			self.word_embedding_size = 32
			self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
			self.text_embedding_size = 128
			self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)
		
		# Resize image embedding
		self.embedding_size = self.semi_memory_size
		if self.use_text:
			self.embedding_size += self.text_embedding_size
		
		# Define actor's model
		self.actor = nn.Sequential(
			nn.Linear(self.embedding_size, 64),
			nn.Tanh(),
			nn.Linear(64, action_space.n)
		)
		
		# Define critic's model
		self.critic = nn.Sequential(
			nn.Linear(self.embedding_size, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)
		
		# Initialize parameters correctly
		self.apply(init_params)
	
	@property
	def memory_size(self):
		return 2 * self.semi_memory_size
	
	@property
	def semi_memory_size(self):
		return self.image_embedding_size
	
	def forward(self, obs, memory, mu_sig_memory):
		x = obs.image.transpose(1, 3).transpose(2, 3)  # [16, 7, 7, 3] -> [16, 3, 7, 7]
		x = self.image_conv(x)  # [16, 7, 7, 3] -> [16, 64, 1, 1]
		x = x.reshape(x.shape[0], -1)  # [16, 64, 1, 1] -> [16, 64]
		
		if self.use_memory:
			hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
			hidden = self.memory_rnn(torch.cat((x, mu_sig_memory), dim=1), hidden)
			embedding = hidden[0]
			memory = torch.cat(hidden, dim=1) ## Here!
		else:
			embedding = x
		
		if self.use_text:
			embed_text = self._get_embed_text(obs.text)
			embedding = torch.cat((embedding, embed_text), dim=1)
		
		y = self.actor(embedding)
		dist = Categorical(logits=F.log_softmax(y, dim=1))
		
		y = self.critic(embedding)
		value = y.squeeze(1)
		
		return dist, value, memory, x
	
	def _get_embed_text(self, text):
		_, hidden = self.text_rnn(self.word_embedding(text))
		return hidden[-1]


class BLModel_Block(nn.Module):
	def __init__(self, obs_space, top_k, block_hid_dim, att_layer_num, norm_type, device, bl_log_sig_min, bl_log_sig_max):
		super().__init__()
		
		n = obs_space["image"][0]
		m = obs_space["image"][1]
		self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64 ## 64
		
		### Block latent variable(embedding) dimension
		self.block_hid_dim = block_hid_dim  # 256
		self.block_mu_size = 64

		### Attention selection
		self.top_k = top_k
		
		self.device = device
		self.bl_log_sig_min = bl_log_sig_min
		self.bl_log_sig_max = bl_log_sig_max
		
		activation = nn.Tanh()
		# activation = nn.ReLU()

		# Define block memory
		## Blockwise RNN. For simplicity, we use GRU. However, LSTM can also be applied.
		self.block_memory_rnn = nn.GRUCell(self.image_embedding_size*self.top_k, self.block_hid_dim)
		
		## Define Self-attention
		## Note that the output dimension of self-attention is the same as input dimension due to Residual connection.
		## 4 multi-heads
		self.att_model_q = BaseModel(hidden_dim=self.image_embedding_size, num_layers=att_layer_num, norm_type=norm_type)
		
		## Define mean of encoder
		self.block_mu = nn.Sequential(
			nn.Linear(self.block_hid_dim, self.block_hid_dim //2),
			activation,
			nn.Linear(self.block_hid_dim // 2, self.block_mu_size)
		)

		## Define stddev of decoder
		self.block_sig = nn.Sequential(
			nn.Linear(self.block_hid_dim, self.block_hid_dim // 2),
			activation,
			nn.Linear(self.block_hid_dim // 2, self.block_mu_size),
			# nn.Softplus()
		)
		
		## Define p_theta model for self-normalized importance sampling
		self.self_norm_model = nn.Sequential(
			nn.Linear(self.image_embedding_size*self.top_k + self.block_mu_size, self.image_embedding_size),
			activation,
			nn.Linear(self.image_embedding_size, 1)
		)
		
		# Initialize parameters correctly
		self.apply(init_params)
		
		
	def forward(self, obs_block_ori, block_memory_ori, bl_update_check):
		## obs_block_ori: [ seq=L, batch=16, 64(feature)]
		## block_memory_ori: [16, self.block_hid_dim=256]
		
		obs_block = obs_block_ori[:, bl_update_check, :]
		# print("obs_block", obs_block.shape)
		assert len(obs_block.size()) == 3
		
		block_memory = block_memory_ori[bl_update_check, :]
		# print("block_memory", block_memory.shape)
		assert len(block_memory.size()) == 2
		
		# print("block_memory first", block_memory_ori)

		# Step 1. input obs and action to Attention for model_q.
		# Then select some of them. Default is to choose 2 ends like bi-LSTM (block_len >=2).
		# Later, we are going to choose 'self.trans_select_N' number of outputs based on Attention score.
		
		trans_q_output, attention_matrix = self.att_model_q.forward(obs_block)  # (seq, batch_size=4, 64)
		trans_q_output = trans_q_output.permute(1, 0, 2)  # (batch=4, seq, 256)
		# print("trans_q_output", trans_q_output, trans_q_output.shape) # (batch, seq, hidden_dim=64)
		# print("attention", attention_matrix, attention_matrix.shape) # (batch*n_head, trans_seq, trans_seq)
		
		attention_matrix_align = torch.cat(attention_matrix.split(obs_block.size()[1], dim=0),2)  # (batch, trans_seq, trans_seq*n_head)
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
		
		top_k_index_repeat = top_k_index.unsqueeze(dim=-1).repeat(1, 1, trans_q_output.shape[2])
		# (batch, min(self.top_k, trans_q_output.shape[1]), 256)
		top_k_q_output_selected = torch.gather(trans_q_output, dim=1, index=top_k_index_repeat)  # selected vectors
		# (batch, min(self.top_k, trans_q_output.shape[1]), 256)
		# print("top_k_q_output_selected", top_k_q_output_selected, top_k_q_output_selected.shape)
		
		reshaped = torch.reshape(top_k_q_output_selected, (batch_here, -1))  # (batch, min(self.top_k, trans_q_output.shape[1])*256)
		# print("reshaped init", reshaped, reshaped.shape)
		
		## Padding if necessary
		if seq_len_here < self.top_k:
			# print("Check~~~~~~~~~~~~~~~~~~~~~~~~~~~")
			zero_pad = torch.zeros((batch_here, (self.top_k - seq_len_here) * trans_q_output.shape[2]), device=self.device)
			reshaped = torch.cat((reshaped, zero_pad), dim=-1)  # (batch, self.top_k*256)
		
		# Step 2. Here, 'block_memory' should be changed to block_variable recurrently.
		block_memory = self.block_memory_rnn(reshaped, block_memory)  # (batch, hidden=256)
		
		block_memory_ori[bl_update_check, :] = block_memory
		
		return block_memory_ori, reshaped
	
	def block_mu_sig(self, block_memory):
		sig = torch.exp(self.block_sig(block_memory).clamp(self.bl_log_sig_min, self.bl_log_sig_max))
		
		return torch.cat((self.block_mu(block_memory), sig), dim=-1)
		