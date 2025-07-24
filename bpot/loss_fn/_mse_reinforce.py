import torch


def mse_reward(
		obtained: torch.Tensor,
		expected: torch.Tensor
) -> torch.Tensor:
	expected = expected.to(obtained)
	reward = -(0.5 * (obtained - expected) ** 2).sum()
	return reward
