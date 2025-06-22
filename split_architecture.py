import torch


class SynapseLayer:
	def __init__(
			self,
			num_in: int,
			num_out: int,
			tau: float = 0.99,
			ls_decay: float = 0,
			lr: float = 1e-3,
			device: str = "cuda",
			is_first_layer: bool = False,
	):
		self.num_in = num_in
		self.num_out = num_out
		self.lr = lr
		self.is_first_layer = is_first_layer

		self.weights = torch.randn(num_in, num_out).to(device)

		self.input_trace = torch.zeros(num_in).to(device)
		self.tau = torch.ones(num_in).to(device) * tau

		self.ls = torch.zeros(num_out).to(device)
		self.ls_decay = torch.ones(num_out).to(device) * ls_decay

	def get_states(self) -> dict:
		return vars(self)

	def update_ls(self, learning_signal):
		if learning_signal is not None:
			assert learning_signal.size() == (
				self.num_out,), f"Neuron.update_ls(): Expected {self.num_out} but got {learning_signal.size()}"
			learning_signal = learning_signal.to(self.weights)
			self.ls = self.ls * self.ls_decay + learning_signal

	def backward(self) -> torch.Tensor:
		"""
		Does the backwards pass, calculating all gradients and updating parameters.
		:return: The passed Learning Signal (LS) to the upstream layer, of size [num_in]
		"""
		# trace * tau + value = trace
		# value = trace - trace * tau
		# value = (1 - tau) * trace
		average_input = (1 - self.tau) * self.input_trace

		learning_signal = self.ls.unsqueeze(0)  # [1, out]
		learning_signal = torch.broadcast_to(learning_signal, (self.num_in, self.num_out))  # [in, out]

		input_tensor = average_input.unsqueeze(-1)  # [in, 1]
		input_tensor = torch.broadcast_to(input_tensor, (self.num_in, self.num_out))  # [in, out]

		delta_w = learning_signal * input_tensor  # d_ls/d_w = input
		delta_input = learning_signal * self.weights  # d_ls/d_in = weight

		self.weights -= delta_w * self.lr  # goes down the gradient

		passed_learning_signal = delta_input.mean(dim=-1)

		return passed_learning_signal

	def forward(self, in_spikes: torch.Tensor, ):
		"""
		Does the forward and backward pass, updating parameters.
		:param in_spikes: input tensor of size [num_in], expected to be spikes
		:return: out_current (to be passed to the downstream layer), passed_learning_signal (to be passed to the upstream layer)
		"""
		assert in_spikes.size() == (
			self.num_in,), f"Synapse.forward(): Expected {self.num_in} but got {in_spikes.size()}"
		in_spikes = in_spikes.to(self.weights)

		passed_learning_signal = self.backward()
		if self.is_first_layer:
			passed_learning_signal = None

		self.input_trace = self.input_trace * self.tau + in_spikes

		out_current = in_spikes @ self.weights

		return out_current, passed_learning_signal


class NeuronLayer:
	def __init__(
			self,
			num_neurons: int,
			mem_decay: float = 0.99,
			threshold: float = 1,
			tau: float = 0.99,
			ls_decay: float = 0,
			lr: float = 1e-3,
			device: str = "cuda",
			is_first_layer: bool = False,
	):
		self.num_neurons = num_neurons
		self.lr = lr
		self.is_first_layer = is_first_layer

		self.mem = torch.zeros(num_neurons).to(device)
		self.mem_decay = torch.ones(num_neurons).to(device) * mem_decay
		self.threshold = torch.ones(num_neurons).to(device) * threshold

		self.input_trace = torch.zeros(num_neurons).to(device)
		self.tau = torch.ones(num_neurons).to(device) * tau

		self.ls = torch.zeros(num_neurons).to(device)
		self.ls_decay = torch.ones(num_neurons).to(device) * ls_decay

	def get_states(self) -> dict:
		return vars(self)

	def update_ls(self, learning_signal):
		if learning_signal is not None:
			assert learning_signal.size() == (
				self.num_neurons,), f"Neuron.update_ls(): Expected {self.num_neurons} but got {learning_signal.size()}"
			learning_signal = learning_signal.to(self.mem)
			self.ls = self.ls * self.ls_decay + learning_signal

	def backward(self):
		"""
		Does the backwards pass, calculating all gradients and updating parameters.
		:return: The passed Learning Signal (LS) to the upstream layer, of size [num_neurons]
		"""
		# trace * tau + value = trace
		# value = trace - trace * tau
		# value = (1 - tau) * trace
		average_input = (1 - self.tau) * self.input_trace

		return average_input

	def forward(self, in_current):
		"""
		Does the forward and backward pass, updating parameters.
		:param in_current: input tensor of size [num_neurons], expected to be the net current to each neuron
		:return: out_spikes (to be passed to the downstream layer), passed_learning_signal (to be passed to the upstream layer)
		"""
		assert in_current.size() == (
			self.num_neurons,), f"Neuron.forward(): Expected {self.num_neurons} but got {in_current.size()}"
		in_current = in_current.to(self.mem)

		passed_learning_signal = self.backward()
		if self.is_first_layer:
			passed_learning_signal = None

		self.mem = self.mem * self.mem_decay + in_current
		out_spikes = (self.mem >= self.threshold).float()
		self.mem -= out_spikes * self.threshold

		return out_spikes, passed_learning_signal
