import torch
from .. import functional


class Softmax:
	def __init__(
			self,
			num_in: int,
			num_out: int,
			mem_decay: float = 1,
			device: str = "cuda",
			lr: float = 1e-2,
	):
		self.num_in = num_in
		self.num_out = num_out
		self.device = device
		self.lr = lr

		self.weight = torch.randn(num_out, num_in).to(device)
		self.raw_mem_decay = functional.sigmoid_inverse((torch.ones(num_out) * mem_decay).to(device))

		self.mem = torch.zeros(num_out).to(device)
		self.in_spikes_trace = torch.zeros(num_in).to(device)
		self.out_spikes_trace = torch.zeros(num_out).to(device)
		self.ls = torch.zeros(num_out).to(device)

		self.in_spikes_trace_decay = torch.ones(num_in).to(device) * 0.9
		self.out_spikes_trace_decay = torch.ones(num_out).to(device) * 0.9

	def calculate_derivatives(self) -> torch.Tensor:
		average_input = functional.decompose_trace(self.in_spikes_trace, self.in_spikes_trace_decay)
		weight_delta = torch.einsum("i, o -> oi", average_input, self.ls)
		learning_signal = torch.einsum("o, oi -> i", self.ls, self.weight)

		self.weight -= weight_delta * self.lr
		self.ls.zero_()

		return learning_signal

	def forward(self, in_spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		in_spikes = in_spikes.to(self.device)
		self.in_spikes_trace = self.in_spikes_trace * self.in_spikes_trace_decay + in_spikes

		learning_signal = self.calculate_derivatives()
		mem_decay = torch.nn.functional.sigmoid(self.raw_mem_decay)

		c = torch.einsum("i, oi -> o", in_spikes, self.weight)
		self.mem = self.mem * mem_decay + c

		distribution_probability = torch.nn.functional.softmax(self.mem, dim=-1)
		index = torch.multinomial(distribution_probability, num_samples=1)
		out_spikes = torch.zeros_like(distribution_probability)
		out_spikes.scatter_(-1, index, 1)

		self.out_spikes_trace = self.out_spikes_trace * self.out_spikes_trace_decay + out_spikes

		return out_spikes, learning_signal

	def backward(self, learning_signal: torch.Tensor) -> None:
		if learning_signal is not None:
			learning_signal = learning_signal.to(self.device)
			self.ls += learning_signal
