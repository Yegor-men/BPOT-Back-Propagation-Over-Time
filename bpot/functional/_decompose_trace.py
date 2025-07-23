import torch


def decompose_trace(
		trace: torch.Tensor,
		trace_decay: torch.Tensor,
) -> torch.Tensor:
	return trace * (1 - trace_decay)
