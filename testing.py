from sympy.series.formal import exp_re

import bpot
import torch

n_in = 100
n_out = 5
n_hidden = 1000

l1 = bpot.layers.Leaky(
	num_in=n_in,
	num_out=n_hidden,
	beta=0.9,
	threshold=1,
)

l2 = bpot.layers.Leaky(
	num_in=n_hidden,
	num_out=n_out,
	beta=0.9,
	threshold=1,
)

rand_in = torch.rand(n_in).round()
rand_out = torch.rand(n_out).round()

def forward():
	x, ls = l1.forward(rand_in)
	x2, ls2 = l2.forward(x)
	l1.backward(ls2)
	loss, ls3 = bpot.loss_fn.mse(x2, rand_out)
	l2.backward(ls3)
	print(f"Loss: {loss}, exp: {rand_out}, rec: {x2}")

for i in range(100):
	forward()