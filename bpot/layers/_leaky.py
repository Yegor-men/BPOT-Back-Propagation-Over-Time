import torch
from .. import functional


class Leaky:
	def __init__(
			self,
			num_in: int,
			num_out: int,
			beta: float = 1,
			threshold: float = 1,
			device: str = "cuda",
			lr: float = 1e-2,
	):
		self.num_in = num_in
		self.num_out = num_out
		self.device = device
		self.lr = lr

		self.weight = torch.randn(num_out, num_in).to(device)
		self.raw_mem_decay = functional.sigmoid_inverse((torch.ones(num_out) * beta).to(device))
		self.raw_threshold = functional.softplus_inverse((torch.ones(num_out) * threshold).to(device))

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
		threshold = torch.nn.functional.softplus(self.raw_threshold)

		c = torch.einsum("i, oi -> o", in_spikes, self.weight)
		self.mem = self.mem * mem_decay + c
		out_spikes = (self.mem >= threshold).float()
		self.mem -= out_spikes * threshold
		self.out_spikes_trace = self.out_spikes_trace * self.out_spikes_trace_decay + out_spikes

		return out_spikes, learning_signal

	def backward(self, learning_signal: torch.Tensor) -> None:
		if learning_signal is not None:
			learning_signal = learning_signal.to(self.device)
			self.ls += learning_signal
