import numpy as np
import random
import torch

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		super(ReplayBuffer, self).__init__()
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim), dtype=np.float32)
		self.action = np.zeros((max_size, action_dim), dtype=np.float32)
		self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
		self.reward = np.zeros((max_size, 1), dtype=np.float32)
		self.not_done = np.zeros((max_size, 1), dtype=np.float32)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		if batch_size > self.size:
			raise ValueError(f"The size of replay buffer must be larger than batch size!!")
		ind = np.random.randint(0, self.size, size=batch_size)
		return (
			torch.as_tensor(self.state[ind],      device=self.device, dtype=torch.float32),
			torch.as_tensor(self.action[ind],     device=self.device, dtype=torch.float32),
			torch.as_tensor(self.next_state[ind], device=self.device, dtype=torch.float32),
			torch.as_tensor(self.reward[ind],     device=self.device, dtype=torch.float32),
			torch.as_tensor(self.not_done[ind],   device=self.device, dtype=torch.float32),
		)

class ValueEstimateReplayBuffer(ReplayBuffer):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		super(ValueEstimateReplayBuffer, self).__init__(\
			state_dim=state_dim, action_dim=action_dim, max_size=max_size)
		self.mjsim = [None] * max_size  # add mjsim for simulating

	def add(self, state, action, next_state, reward, done, mjsim):
		idx = self.ptr
		super().add(state, action, next_state, reward, done)
		self.mjsim[idx] = mjsim

	def sample_sim_state(self, batch_size):
		if batch_size > self.size:
			raise ValueError(f"The size of replay buffer must be larger than batch size!!")
		ind = np.random.randint(0, self.size, size=batch_size)

		mjsim_part = [None] * batch_size
		i = 0
		while i < batch_size:
			warpped_state = {}
			warpped_state["mjsim_state"] = self.mjsim[ind[i]]
			warpped_state["env_state"] = self.state[ind[i]]
			mjsim_part[i] = warpped_state
			i += 1

		return (
			torch.as_tensor(self.state[ind], device=self.device, dtype=torch.float32),
			mjsim_part
		)
