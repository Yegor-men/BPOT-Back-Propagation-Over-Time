import bpot
import torch

num_elements = 10
num_labels = 10

snn = bpot.layers.Sequential(
	bpot.layers.LIF(
		num_in=num_elements,
		num_out=num_elements,
		mem_decay=0.9,
		threshold=1,
		activation_fn="threshold"
	),
	bpot.layers.LIF(
		num_in=num_elements,
		num_out=num_elements,
		mem_decay=0.9,
		threshold=1,
		activation_fn="threshold"
	),
	bpot.layers.LIF(
		num_in=num_elements,
		num_out=num_elements,
		mem_decay=0.9,
		threshold=1,
		activation_fn="threshold"
	),
)

random_in = torch.rand(num_labels).round()
random_out = torch.zeros(num_labels)
random_out[0] = 1

num_steps = 100

model_outputs = []

for step in range(num_steps):
	model_out = snn.forward(random_in)
	loss, loss_derivative = bpot.loss_fn.mse(model_out, random_out)
	snn.backward(loss_derivative)
	print(f"{step} - loss: {loss}")
	model_outputs.append(model_out)

bpot.plot.spike_train(model_outputs)