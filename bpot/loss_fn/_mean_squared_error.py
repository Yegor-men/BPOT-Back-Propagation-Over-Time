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


def mse_index(
		obtained: torch.Tensor,
		index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute the loss and learning signal of a model's outputs by using the mean squared error.

	:param obtained: a tensor of size [n]; the output produced by the model
	:param expected: a tensor of size [n]; the output the model should have produced instead
	:return: tuple: (loss, learning signal)
	"""
	expected = torch.zeros_like(obtained).scatter_(-1, index, 1)
	loss = (0.5 * (obtained - expected) ** 2).sum()
	ls = 2.0 * (obtained - expected)
	return loss, ls


def mse_reward(
		obtained: torch.Tensor,
		expected: torch.Tensor
) -> torch.Tensor:
	expected = expected.to(obtained)
	reward = -(0.5 * (obtained - expected) ** 2).sum()
	return reward
