import torch
import matplotlib.pyplot as plt
import numpy as np


def Spike_Train(list_of_tensors):
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
