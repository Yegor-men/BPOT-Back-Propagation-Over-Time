import torch
import math

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.set_grad_enabled(False)


class Synapse:
	def __init__(
			self,
			num_in: int,
			num_out: int,
			device: str = "cuda",
			tau: float = 0.99,
			lr: float = 1e-3,
			is_first_layer: bool = False,
	):
		self.num_in = num_in
		self.num_out = num_out
		self.lr = lr
		self.tau = tau
		self.is_first_layer = is_first_layer

		self.input_trace = torch.zeros(num_in).to(device)
		self.ls = torch.zeros(num_out).to(device)

		self.w = torch.randn(num_in, num_out).to(device)

	def update_ls(self, ls: torch.Tensor):
		if ls is not None:
			assert ls.size() == (self.num_out,), "Synapse.update_ls: tensor size mismatch"
			ls = ls.to(self.w)
			self.ls += ls

	def backward(self):
		if self.is_first_layer:
			return None

		average_inputs = self.input_trace * (1 - self.tau)

		ls = self.ls.unsqueeze(0)  # [1, out]
		ls = torch.broadcast_to(ls, (self.num_in, self.num_out))  # [in, out]

		average_inputs = average_inputs.unsqueeze(-1)  # [in, 1]
		average_inputs = torch.broadcast_to(average_inputs, (self.num_in, self.num_out))  # [in, out]

		dls_dw = ls * average_inputs  # d_ls/d_w = input
		dls_di = ls * self.w  # d_ls/d_in = weight

		self.w -= dls_dw * self.lr  # goes down the gradient

		passed_learning_signal = dls_di.mean(dim=-1)

		self.ls *= self.tau

		return passed_learning_signal

	def forward(self, in_spikes: torch.Tensor):
		assert in_spikes.size() == (self.num_in,), "Synapse.forward: tensor size mismatch"
		in_spikes = in_spikes.to(self.w)

		self.input_trace = self.input_trace * self.tau + in_spikes

		passed_ls = self.backward()

		net_current = in_spikes @ self.w

		return net_current, passed_ls


class Neuron:
	def __init__(
			self,
			num_in: int,
			activation_fn: str = "threshold",
			mem_decay: float = 0.99,
			threshold: float = 1,
			device: str = "cuda",
			tau: float = 0.99,
			lr: float = 1e-3,
			is_first_layer: bool = False,
	):
		valid_output_types = ["threshold", "softmax"]
		if activation_fn not in valid_output_types:
			raise ValueError(f"output type must be in {valid_output_types}")
		self.output_type = activation_fn

		self.num_in = num_in
		self.lr = lr
		self.tau = tau
		self.is_first_layer = is_first_layer

		self.input_trace = torch.zeros(num_in).to(device)
		self.ls = torch.zeros(num_in).to(device)

		self.mem = torch.zeros(num_in).to(device)
		self.threshold = (torch.ones(num_in) * torch.log(torch.Tensor([torch.e ** threshold - 1]))).to(device)
		self.mem_decay = (torch.ones(num_in) * torch.log(torch.Tensor([mem_decay / (1 - mem_decay)]))).to(device)
		approx_thresh = torch.nn.functional.softplus(self.threshold[0]).item()
		approx_mem_decay = torch.nn.functional.sigmoid(self.mem_decay[0]).item()
		assert math.isclose(approx_thresh, threshold, rel_tol=1e-3), "Threshold inverse function didn't work"
		assert math.isclose(approx_mem_decay, mem_decay, rel_tol=1e-3), "Mem decay inverse function didn't work"

	def update_ls(self, ls: torch.Tensor):
		if ls is not None:
			assert ls.size() == (self.num_in,), "Neuron.update_ls: tensor size mismatch"
			ls = ls.to(self.mem)
			self.ls += ls

	def backward(self):
		if self.is_first_layer:
			return None

		passed_ls = self.ls * torch.sign(self.input_trace)

		return passed_ls

	def forward(self, in_current: torch.Tensor):
		assert in_current.size() == (self.num_in,), "Neuron.forward: tensor size mismatch"
		in_current = in_current.to(self.mem)

		self.input_trace = self.input_trace * self.tau + in_current

		passed_ls = self.backward()

		mem_decay = torch.nn.functional.sigmoid(self.mem_decay)
		self.mem = self.mem * mem_decay + in_current

		if self.output_type == "threshold":
			threshold = torch.nn.functional.softplus(self.threshold)
			out_spikes = self.mem >= threshold
			self.mem -= out_spikes * threshold
		if self.output_type == "softmax":
			distribution_probability = torch.nn.functional.softmax(self.mem, dim=-1)
			index = torch.multinomial(distribution_probability, num_samples=1)
			out_spikes = torch.zeros_like(distribution_probability)
			out_spikes.scatter_(-1, index, 1)

		return out_spikes, passed_ls


class SNN:
	def __init__(
			self,
			num_in: int,
			num_out: int,
			hidden_layer_size: int,
			num_hidden_layers: int,
			use_softmax: bool = True,
			lr: float = 1e-3,
	):
		self.correct_layers = [hidden_layer_size for i in range(num_hidden_layers * 2)]
		self.correct_layers[0], self.correct_layers[-1] = num_in, num_out

		self.layers = []

		for i in range(len(self.correct_layers) - 1):
			synapse = Synapse(num_in=self.correct_layers[i], num_out=self.correct_layers[i + 1], lr=lr)
			neuron = Neuron(num_in=self.correct_layers[i + 1], lr=lr)
			self.layers.append(synapse)
			self.layers.append(neuron)

		if use_softmax:
			self.layers[-1] = Neuron(num_in=self.correct_layers[-1], activation_fn="softmax", lr=lr)

		self.layers[0].is_first_layer = True

	def calculate_loss(self, model_output, desired_output):
		loss = torch.sum((model_output - desired_output) ** 2).item()

	def forward(self, x: torch.Tensor):
		for index, layer in enumerate(self.layers):
			x, ls = layer.forward(x)
			self.layers[index - 1].update_ls(ls)

		return x


rand_size = 10
rand_input = torch.rand(rand_size).round()
rand_output = torch.rand(rand_size).round()
hidden_size = int(rand_size * 2)
num_layers = 3

snn = SNN(
	num_in=rand_size,
	num_out=rand_size,
	hidden_layer_size=hidden_size,
	num_hidden_layers=num_layers,
)

model_out = snn.forward(rand_input)
print(f"Input: {rand_input}")
print(f"Model out: {model_out}")
print(f"Expected output: {rand_output}")
