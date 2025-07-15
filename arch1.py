import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.set_grad_enabled(False)


class LIF:
	def __init__(
			self,
			num_in: int,
			num_out: int,
			mem_decay: float,
			threshold: float,
			activation_fn: str,
			device: str = "cuda",
			lr: float = 1e-2,
	):
		activation_fns = ["threshold", "softmax"]
		assert activation_fn in activation_fns, f"activation_fn not in {activation_fns}"

		self.num_in = num_in
		self.num_out = num_out
		self.activation_fn = activation_fn
		self.device = device
		self.lr = lr

		# sigmoid = 1/(1+e^-x), inverse = ln(y/(1-y))
		# softplus = ln(1+e^x), inverse = ln(e^y-1)
		self.weight = torch.randn(num_out, num_in).to(device)
		self.raw_mem_decay = (torch.ones(num_out) * torch.log(torch.tensor([mem_decay / (1 - mem_decay)]))).to(device)
		self.raw_threshold = (torch.ones(num_out) * torch.log(torch.tensor([torch.e ** threshold - 1]))).to(device)

		import math

		approx_mem_decay = torch.nn.functional.sigmoid(self.raw_mem_decay[0]).item()
		approx_thresh = torch.nn.functional.softplus(self.raw_threshold[0]).item()
		assert math.isclose(approx_mem_decay, mem_decay, rel_tol=1e-3), "Mem decay inverse function didn't work"
		assert math.isclose(approx_thresh, threshold, rel_tol=1e-3), "Threshold inverse function didn't work"

		self.mem = torch.zeros(num_out).to(device)
		self.input_trace = torch.zeros(num_in).to(device)
		self.input_decay = torch.ones(num_in).to(device) * 0.9
		self.ls = torch.zeros(num_out).to(device)

	def update_ls(self, learning_signal):
		if learning_signal is not None:
			assert learning_signal.size() == (self.num_out,), "learning_signal wrong size"
			learning_signal = learning_signal.to(self.device)
			self.ls += learning_signal

	def forward(self, in_spikes):
		assert in_spikes.size() == (self.num_in,), "in_spikes wrong size"
		in_spikes = in_spikes.to(self.device)

		self.input_trace = self.input_trace * self.input_decay + in_spikes

		current = in_spikes @ self.weight.T

		mem_decay = torch.nn.functional.sigmoid(self.raw_mem_decay)
		threshold = torch.nn.functional.softplus(self.raw_threshold)

		self.mem = self.mem * mem_decay + current

		if self.activation_fn == "threshold":
			out_spikes = (self.mem >= threshold).float()
			self.mem = self.mem - threshold * out_spikes
		elif self.activation_fn == "softmax":
			distribution_probability = torch.nn.functional.softmax(self.mem, dim=-1)
			index = torch.multinomial(distribution_probability, num_samples=1)
			out_spikes = torch.zeros_like(distribution_probability)
			out_spikes.scatter_(-1, index, 1)
			out_spikes = out_spikes.float()
		else:
			out_spikes = None

		return out_spikes

	def backward(self):
		average_input = self.input_trace * (1 - self.input_decay)
		weight_delta = torch.einsum("i,o->oi", average_input, self.ls)
		learning_signal = torch.einsum("o,oi->i", self.ls, self.weight)

		self.weight -= weight_delta * self.lr
		self.ls.zero_()

		return learning_signal


class Sequential:
	def __init__(self, *layers):
		self.layers = list(layers)

	def forward(self, x: torch.Tensor):
		for index, layer in enumerate(self.layers):
			x = layer.forward(x)
			ls = layer.backward() if index != 0 else None
			self.layers[index - 1].update_ls(ls)
		return x

	def backward(self, ls: torch.Tensor):
		self.layers[-1].update_ls(ls)


def mse_loss(received, expected):
	received, expected = received.cpu(), expected.cpu()
	loss = (0.5 * (received - expected) ** 2).sum().item()
	ls = 2.0 * (received - expected)
	return loss, ls


def jensen_shannon_loss(received, expected):
	received, expected = received.cpu(), expected.cpu()
	P, Q, M = expected, received, 0.5 * (expected + received)
	loss = (0.5 * (P * torch.log(P / M)).sum() + 0.5 * (Q * torch.log(Q / M)).sum()).item()
	ls = 0.5 * torch.log(Q / M)
	return loss, ls


num_elements = 10
num_labels = 10

snn = Sequential(
	LIF(
		num_in=num_elements,
		num_out=num_elements,
		mem_decay=0.9,
		threshold=1,
		activation_fn="threshold"
	),
	LIF(
		num_in=num_elements,
		num_out=num_elements,
		mem_decay=0.9,
		threshold=1,
		activation_fn="threshold"
	),
	LIF(
		num_in=num_elements,
		num_out=num_labels,
		mem_decay=0.9,
		threshold=1,
		activation_fn="softmax"
	),
)

random_in = torch.rand(num_labels).round()
random_out = torch.zeros(num_labels)
random_out[0] = 1

num_steps = 100

model_outputs = []

for step in range(num_steps):
	model_out = snn.forward(random_in)
	loss, loss_derivative = mse_loss(model_out, random_out)
	snn.backward(loss_derivative)
	print(f"{step} - loss: {loss}")
	model_outputs.append(model_out)


def plot_spike_train(list_of_tensors):
	import matplotlib.pyplot as plt
	import numpy as np

	num_timesteps = len(list_of_tensors)
	num_neurons = list_of_tensors[0].numel()

	xs = []
	ys = []
	for t, vec in enumerate(list_of_tensors):
		arr = vec.cpu().numpy().ravel()
		spike_indices = np.nonzero(arr)[0]
		xs.extend([t] * len(spike_indices))
		ys.extend(spike_indices.tolist())

	plt.figure()
	plt.scatter(xs, ys, marker='|')
	plt.xlabel("Time step")
	plt.ylabel("Element index")
	plt.title("Spike Train Raster")

	plt.yticks(range(num_neurons))
	# plt.xticks(range(num_timesteps))

	plt.show()


plot_spike_train(model_outputs)
