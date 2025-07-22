import torch


def mse(
		obtained: torch.Tensor,
		expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute the loss and learning signal of a model's outputs by using the mean squared error.

	:param obtained: a tensor of size [n]; the output produced by the model
	:param expected: a tensor of size [n]; the output the model should have produced instead
	:return: tuple: (loss, learning signal)
	"""
	expected = expected.to(obtained)
	loss = (0.5 * (obtained - expected) ** 2).sum()
	ls = 2.0 * (obtained - expected)
	return loss, ls
