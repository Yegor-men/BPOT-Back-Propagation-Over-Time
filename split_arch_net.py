from split_architecture import SynapseLayer as SL
from split_architecture import NeuronLayer as NL
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.set_grad_enabled(False)

lr = 1e-1


class Network:
	def __init__(self):
		self.layers = []

		for _ in range(3):
			sl = SL(num_in=10, num_out=10, lr=lr)
			nl = NL(num_neurons=10, lr=lr)
			self.layers.append(sl)
			self.layers.append(nl)

		self.layers[0].is_first_layer = True

	def forward(self, tensor):
		for index, layer in enumerate(self.layers):
			tensor, ls = layer.forward(tensor)
			self.layers[index - 1].update_ls(ls)
		return tensor


net = Network()
rand_in = torch.rand(10).round()
rand_out = torch.rand(10).round()

spike_train = []
for i in range(100):
	out = net.forward(rand_in)
	spike_train.append(out.cpu())
	loss = rand_out - out.cpu()
	net.layers[-1].update_ls(loss)

print(f"IN: {rand_in}")
print(f"OUT: {out}")
print(f"EXP OUT: {rand_out}")
print(f"LOSS: {loss}")

for index, layer in enumerate(net.layers):
	params = layer.get_states()
	print(f"Layer {index} - LS: {params['ls']}")


def plot_spike_train(spike_train_list):
	import matplotlib.pyplot as plt
	outputs_tensor = torch.stack(spike_train_list)
	out_size = len(spike_train_list[0])
	num_timesteps = len(spike_train_list)
	spike_times = []
	for neuron_index in range(out_size):
		times = [t for t in range(num_timesteps) if outputs_tensor[t, neuron_index] == 1]
		spike_times.append(times)

	plt.eventplot(spike_times, lineoffsets=range(out_size), linelengths=0.8, color="black")

	plt.yticks(range(out_size), [f"Neuron {i}" for i in range(out_size)])
	plt.xlabel("Timestep")
	plt.ylabel("Neuron")
	plt.title(f"Spike Train Visualization Over {num_timesteps:,} Timesteps")
	plt.tight_layout()
	plt.show()


plot_spike_train(spike_train)
