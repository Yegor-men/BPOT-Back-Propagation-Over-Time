import torch


class Layer:
    def __init__(
        self,
        *,
        num_in: int,
        num_out: int,
        threshold: float = 1,
        mem_decay: float = 0.99,
        ls_decay: float = 0,
        device: str = "cuda",
        lr: float = 1e-3,
    ) -> None:

        self.ls = torch.zeros(num_out).to(device)
        self.ls_decay = torch.ones(num_out).to(device) * ls_decay
        # Not sure if I want this as a learnable, keep it a tensor for now

        self.w = torch.randn(num_in, num_out).to(device) * 0.1
        self.mem = torch.zeros(num_out).to(device)
        self.mem_decay = torch.ones(num_out).to(device) * mem_decay
        self.threshold = torch.ones(num_out).to(device) * threshold
        # Threshold is nicer as positive, so we pass it through softplus before use

        self.input_trace = torch.zeros(num_in).to(device)
        self.output_trace = torch.zeros().to(device)
        # The output trace has to use mem_decay
        # The input trace should ideally use the previous layer's mem_decay
        # Which also implies that the learning signal passed to the previous layer depends on the previous layer's mem_decay
        # Which implies that when this layer gets a learning signal, it's based on this layer's mem_decay
        # Hence there's a relation between output trace and the learning signal?
        # TODO figure this out

    def update_ls(
        self,
        incoming_ls: torch.Tensor,
    ) -> None:
        self.ls *= self.ls_decay
        # I don't think that having a trace system for the learning signal does any good? But I'll keep it here just in case
        self.ls += incoming_ls

    def calculate_derivatives(
        self,
    ):
        # surrogate derivative can be based on either arctan or sigmoid
        # sigmoid is easiest as it's built in, also the derivative is easy as hell
        # s(x) = e^x / (1 + e^x)
        # s'(x) = [s(x)][1 - s(x)]

        # "firing" is approximated via the sigmoid of the mem level
        # round(s(mem - threshold)) = {0, 1}
        # but this is actually mem_new, not mem
        # mem_new = mem_old * mem_decay + current
        # but current is matrix multiplication
        # for the sake of simplicity, current is in_something * w_something = i*w

        # The incoming LS is the d_loss/d_o, the derivative of the loss with respect to this layer's output

        # need to calculate the derivative of all learnable parameters
        # d_w, d_mem_decay, d_thresh
        # also need to calculate the derivative with respect to the input to pass as the learning signal to the previous layer
        # d_i
        
        # Hence, the output function can be calculated like this:
        # o(mem, t) = e^(mem - t) / (1 + e^(mem - t))
        # o(mem_old, mem_decay, t) = e^(mem_old * mem_decay + current - t) / (1 + e^(mem_old * mem_decay + current - t))
        # o(mem_old, mem_decay, t, i, w) = e^(mem_old * mem_decay + i*w - t) / (1 + e^(mem_old * mem_decay + i*w - t))
        

        pass

    def update_parameters(
        self,
    ):
        pass

    def forward(
        self,
        in_spikes: torch.Tensor,
    ):
        self.calculate_derivatives()
        self.update_parameters()

        raw_current = in_spikes @ self.w
        self.mem *= self.mem_decay
        self.mem += raw_current

        softplus_thresh = torch.nn.functional.softplus(self.threshold)
        out_spikes = (self.mem >= softplus_thresh).float()
        self.mem -= out_spikes * softplus_thresh

        return out_spikes
