import bpot
import torch

n_in = 1000
n_out = 10
n_hidden = 1000

model = bpot.layers.Sequential(
	bpot.layers.Leaky(
		num_in=n_in,
		num_out=n_hidden,
	),
	bpot.layers.Leaky(
		num_in=n_hidden,
		num_out=n_out,
	)
)

rand_in = torch.rand(n_in).round()
rand_out = torch.rand(n_out).round()

model_outputs = []

for i in range(100):
	model_out = model.forward(rand_in)
	loss, learning_signal = bpot.loss_fn.mse(model_out, rand_out)
	model.backward(learning_signal)
	print(f"{i:,} - loss: {loss.item()}")
	model_outputs.append(model_out)

print(f"exp: {rand_out}\ngot: {model_outputs[-1]}")
bpot.plot.spike_train(model_outputs)