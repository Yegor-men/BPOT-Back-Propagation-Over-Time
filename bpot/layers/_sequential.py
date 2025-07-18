import torch


class Sequential:
	def __init__(self, *layers):
		self.layers = list(layers)

	def forward(self, x: torch.Tensor):
		for index, layer in enumerate(self.layers):
			x, ls = layer.forward(x)
			if index == 0:
				ls = None
			self.layers[index - 1].backward(ls)
		return x

	def backward(self, ls: torch.Tensor):
		self.layers[-1].backward(ls)
