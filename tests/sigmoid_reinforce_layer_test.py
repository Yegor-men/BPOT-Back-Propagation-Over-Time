import torch, bpot

n_in = 4
n_out = 10
n_hidden = 5
n_timesteps = 1000

model = bpot.layers.Sequential(
	bpot.layers.Leaky(
		num_in=n_in,
		num_out=n_hidden,
	),
	bpot.layers.SigmoidReinforce(
		num_in=n_hidden,
		num_out=n_out,
	)
)

rand_in = torch.rand(n_in).round()
rand_out = torch.rand(n_out).round()

model_outputs = []

for i in range(n_timesteps):
	model_out = model.forward(rand_in)
	reward = bpot.loss_fn.mse_reward(model_out, rand_out)
	print(reward)
	model.backward(-reward)
	print(f"{i:,} - loss: {reward.item()}")
	model_outputs.append(model_out)

print(f"exp: {rand_out}\ngot: {model_outputs[-1]}")
bpot.plot.spike_train(model_outputs)
