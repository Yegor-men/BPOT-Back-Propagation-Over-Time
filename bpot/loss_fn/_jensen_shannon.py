import torch


def js(received, expected):
	"""
	Applies the Jensen Shannon loss function
	:param received: The probability distribution the model produced
	:param expected: The probability distribution the model should've produced
	:return:
	"""
	received, expected = received.cpu(), expected.cpu()
	P, Q, M = expected, received, 0.5 * (expected + received)
	loss = (0.5 * (P * torch.log(P / M)).sum() + 0.5 * (Q * torch.log(Q / M)).sum()).item()
	ls = 0.5 * torch.log(Q / M)
	return loss, ls
