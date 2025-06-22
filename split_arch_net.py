from split_architecture import SynapseLayer as SL
from split_architecture import NeuronLayer as NL
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.set_grad_enabled(False)


class Network:
	def __init__(
			self,
	):
		self.layers = []

		for _ in range(3):
			sl = SL(num_in=10, num_out=10)
			nl = NL(num_neurons=10)
			self.layers.append(sl)
			self.layers.append(nl)

		self.layers[0] = SL(num_in=10, num_out=10, is_first_layer=True)

	def forward(self, tensor):
		for index, layer in enumerate(self.layers):
			tensor, ls = layer.forward(tensor)
			self.layers[index - 1].update_ls(ls)
		return tensor


net = Network()
rand_in = torch.rand(10).round()
rand_out = torch.rand(10).round()
for i in range(100):
	out = net.forward(rand_in)
	loss = rand_out - out.cpu()
	net.layers[-1].update_ls(loss)

print(f"IN: {rand_in}")
print(f"OUT: {out}")
print(f"EXP OUT: {rand_out}")
print(f"LOSS: {loss}")

for index, layer in enumerate(net.layers):
	params = layer.get_states()
	print(f"Layer {index} - LS: {params['ls']}")
