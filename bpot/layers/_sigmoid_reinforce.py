import torch
from .. import functional


class SigmoidReinforce:
	def __init__(
			self,
			num_in: int,
			num_out: int,
			mem_decay: float = 1,
			threshold: float = 1,
			device: str = "cuda",
			lr: float = 1e-2,
	):
		self.num_in = num_in
		self.num_out = num_out
		self.device = device
		self.lr = lr

		self.weight = torch.randn(num_out, num_in).to(device)
		self.raw_mem_decay = functional.sigmoid_inverse((torch.ones(num_out) * mem_decay).to(device))
		self.raw_threshold = functional.softplus_inverse((torch.ones(num_out) * threshold).to(device))

		self.mem = torch.zeros(num_out).to(device)
		self.reward = torch.zeros(1).to(device)

		self.in_spikes_trace = torch.zeros(num_in).to(device)
		self.prob_dist_trace = torch.zeros(num_out).to(device)
		self.out_spikes_trace = torch.zeros(num_out).to(device)

		self.in_spikes_trace_decay = torch.ones(num_in).to(device) * 0.9
		self.prob_dist_trace_decay = torch.ones(num_out).to(device) * 0.9
		self.out_spikes_trace_decay = torch.ones(num_out).to(device) * 0.9

	def calculate_derivatives(self) -> torch.Tensor:
		average_prob_dist = functional.decompose_trace(self.prob_dist_trace, self.prob_dist_trace_decay)
		average_out_spikes = functional.decompose_trace(self.out_spikes_trace, self.out_spikes_trace_decay)
		avg_prob_choice = average_prob_dist * average_out_spikes + (1 - average_prob_dist) * (1-average_out_spikes)

		self_ls = -torch.log(avg_prob_choice) * self.reward
		average_input = functional.decompose_trace(self.in_spikes_trace, self.in_spikes_trace_decay)
		weight_delta = torch.einsum("i, o -> oi", average_input, self_ls)
		learning_signal = torch.einsum("o, oi -> i", self_ls, self.weight)

		self.weight -= weight_delta * self.lr
		self.reward.zero_()

		return learning_signal

	def forward(self, in_spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		in_spikes = in_spikes.to(self.device)
		self.in_spikes_trace = self.in_spikes_trace * self.in_spikes_trace_decay + in_spikes

		learning_signal = self.calculate_derivatives()

		mem_decay = torch.nn.functional.sigmoid(self.raw_mem_decay)
		threshold = torch.nn.functional.softplus(self.raw_threshold)
		self.mem = self.mem * mem_decay + torch.einsum("i, oi -> o", in_spikes, self.weight)
		probability_distribution = torch.nn.functional.sigmoid(self.mem - threshold)
		out_spikes = torch.bernoulli(probability_distribution)

		self.prob_dist_trace *= self.prob_dist_trace_decay
		self.prob_dist_trace += probability_distribution
		self.out_spikes_trace *= self.out_spikes_trace_decay
		self.out_spikes_trace += out_spikes

		return out_spikes, learning_signal

	def backward(self, reward: torch.Tensor) -> None:
		if reward is not None:
			reward = reward.to(self.device)
			self.reward += reward
