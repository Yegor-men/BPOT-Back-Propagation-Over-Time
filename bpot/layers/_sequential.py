import torch


class Sequential:
	def __init__(self, *layers):
		self.layers = list(layers)

	def forward(self, x: torch.Tensor):
		for index, layer in enumerate(self.layers):
			x = layer.forward(x)
			ls = layer.backward() if index != 0 else None
			self.layers[index - 1].update_ls(ls)
		return x

	def backward(self, ls: torch.Tensor):
		self.layers[-1].update_ls(ls)
