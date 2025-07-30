import torch


class Sequential:
	def __init__(self, *layers) -> None:
		self.layers = list(layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for index, layer in enumerate(self.layers):
			x = layer.forward(x)
		return x

	def backward(self, ls: torch.Tensor) -> None:
		for index, layer in enumerate(reversed(self.layers)):
			ls = layer.backward(ls)


class LIF:
	def __init__(
			self,
			n_in: int,
			n_out: int,
			device: str = "cuda",
	) -> None:
		self.weight = torch.randn(n_out, n_in, requires_grad=True).to(device)
		self.mem_decay = torch.randn(n_out, requires_grad=True).to(device)
		self.threshold = torch.randn(n_out, requires_grad=True).to(device)

		self.mem = torch.zeros(n_out).to(device)
		self.in_trace = torch.zeros(n_in).to(device)
		self.in_trace_decay = 0.9

	def forward(self, in_spikes: torch.Tensor) -> torch.Tensor:
		in_spikes = in_spikes.to(self.weight)

		mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
		threshold = torch.nn.functional.softplus(self.threshold)

		self.mem = self.mem * mem_decay + torch.einsum("i, oi -> o", in_spikes, self.weight)
		out_spikes = (self.mem >= threshold).float()
		self.mem -= out_spikes * threshold

		return out_spikes

	def backward(self, learning_signal) -> torch.Tensor:
		average_input = self.in_trace * (1 - self.in_trace_decay)
		average_input.requires_grad_()

		average_current = torch.einsum("i, oi -> o", average_input, self.weight)

		mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
		threshold = torch.nn.functional.softplus(self.threshold)

		# if i=t(1-d) for trace decomposition then t=i/(1-d)
		average_mem = average_current / (1 - mem_decay)

		average_output = torch.nn.functional.sigmoid(average_mem - threshold)
		average_output.backward(learning_signal)

		self.weight -= self.weight.grad
		self.mem_decay -= self.mem_decay.grad
		self.threshold -= self.threshold.grad

		# TODO reset all the .grad states for the learnable parameters
		# TODO maybe not even use .backward but instead go through torch.autograd.grad? This seems fumbly idk
		# TODO somehow make use of global lr for parameter updating, is it possible to somehow use the torch optimizer?

		passed_ls = average_input.grad

		return passed_ls
