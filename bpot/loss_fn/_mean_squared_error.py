import torch


def mse(received: torch.Tensor, expected: torch.Tensor):
	"""
	Applies the mean squared error loss based on the received model output vs the desired model output
	:param received: The actual model's output.
	:param expected: The output the model should've returned
	:return: int(loss), torch.Tensor(learning_signal)
	"""
	received, expected = received.cpu(), expected.cpu()
	loss = (0.5 * (received - expected) ** 2).sum().item()
	ls = 2.0 * (received - expected)
	return loss, ls
