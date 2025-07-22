import torch


def jsd(
		obtained: torch.Tensor,
		expected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute the loss and learning signal of a model's outputted probability distribution by using the Jensen Shannon divergence function.

	:param obtained: a tensor of size [n]; the probability distribution produced by the model
	:param expected: a tensor of size [n]; the probability distribution the model should have produced instead
	:return: tuple: (loss, learning signal)
	"""
	expected = expected.to(obtained)
	P, Q, M = expected, obtained, 0.5 * (expected + obtained)
	loss = (0.5 * (P * torch.log(P / M)).sum() + 0.5 * (Q * torch.log(Q / M)).sum())
	ls = 0.5 * torch.log(Q / M)
	return loss, ls
