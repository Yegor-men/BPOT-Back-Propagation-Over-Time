import torch
import matplotlib.pyplot as plt
import numpy as np


def continuous_train(list_of_tensors: list[torch.Tensor]) -> None:
	T = len(list_of_tensors)
	N = list_of_tensors[0].numel()

	data = torch.stack(list_of_tensors).cpu().numpy()  # → shape [T, N]

	time = range(T)
	plt.figure(figsize=(8, 4))
	for neuron_idx in range(N):
		plt.plot(time, data[:, neuron_idx], label=f'Neuron {neuron_idx}')
	plt.xlabel('Time step')
	plt.ylabel('Membrane potential')
	plt.title('Membrane potential of each neuron over time')
	plt.legend(loc='upper right', ncol=2, fontsize='small')
	plt.tight_layout()
	plt.show()