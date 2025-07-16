import torch


def MSE(received, expected):
	received, expected = received.cpu(), expected.cpu()
	loss = (0.5 * (received - expected) ** 2).sum().item()
	ls = 2.0 * (received - expected)
	return loss, ls
