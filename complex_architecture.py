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
        empty_layer: bool = False,
    ) -> None:
        self.empty = empty_layer
        self.num_in = num_in
        self.num_out = num_out
        self.lr = lr

        self.ls = torch.zeros(num_out).to(device)
        self.ls_decay = torch.ones(num_out).to(device) * ls_decay
        # Not sure if I want this as a learnable, keep it a tensor for now

        self.w = torch.randn(num_in, num_out).to(device) * 0.1
        self.mem = torch.zeros(num_out).to(device)
        self.mem_decay = torch.ones(num_out).to(device) * mem_decay
        self.threshold = torch.ones(num_out).to(device) * threshold
        # Threshold is nicer as positive, so we pass it through softplus before use
        # Derivative of softplus is sigmoid, how convenient :)

        self.input_trace = torch.zeros(num_in).to(device)
        self.output_trace = torch.zeros().to(device)
        # The output trace has to use mem_decay
        # The input trace should ideally use the previous layer's mem_decay

    def get_output_trace(
        self,
    ) -> torch.Tensor:
        return self.output_trace

    def update_ls(
        self,
        incoming_ls: torch.Tensor,
    ) -> None:
        self.ls *= self.ls_decay
        # I don't think that having a trace system for the learning signal does any good? But I'll keep it here just in case
        self.ls += incoming_ls

    def calculate_derivatives(
        self,
        prev_layer,
    ) -> torch.Tensor:
        # surrogate derivative can be based on either arctan or sigmoid
        # sigmoid is easiest as it's built in, also the derivative is easy as hell
        # s(x) = e^x / (1 + e^x)
        # s'(x) = [s(x)][1 - s(x)]
        # s'(x) = 1 / (1 + e^(-x))

        # "firing" is approximated via the sigmoid of the mem level
        # round(s(mem - threshold)) = {0, 1}
        # but this is actually mem_new, not mem
        # mem_new = mem_old * mem_decay + current
        # but current is matrix multiplication
        # for the sake of simplicity, current is in_something * w_something = i*w

        # The incoming LS is the d_loss/d_o, the derivative of the loss with respect to this layer's output

        # need to calculate the derivative of all learnable parameters
        # d_w, d_mem_decay, d_thresh
        # also need to calculate the derivative with respect to the input to pass as the learning signal to the upstream layer
        # d_i

        # Hence, the output function can be calculated like this:
        # o(mem, t) = e^(mem - t) / (1 + e^(mem - t))
        # o(mem_old, mem_decay, t) = e^(mem_old * mem_decay + current - t) / (1 + e^(mem_old * mem_decay + current - t))
        # o(mem_old, mem_decay, t, i, w) = e^(mem_old * mem_decay + i*w - t) / (1 + e^(mem_old * mem_decay + i*w - t))
        # Say that x = mem_old * mem_decay + i*w - t, in that case:
        # o(x) = e^x / (1 + e^x)
        # o'(x) = d_o/d_x = [o(x)][1 - o(x)] = 1 / (1 + o^(-x)) = SOME NUMBER
        # We want to use the chain rule, so we already have a number for d_loss/d_o, we just did d_o/dx,
        # We know that x = mem_old * mem_decay + i*w - t
        # So now we need d_x with respect to mem_decay, i, w, t
        # d_x/d_mem_decay = mem_old
        # d_x/d_i = w
        # d_x/d_w = i
        # d_x/d_t = derivative of softplus = sigmoid(t) = S(t) * -1

        # Therefore, we have:
        # Gradient of w: d_loss/d_w = (d_loss/d_o) * (d_o/d_x) * (d_x/d_w) = LS * S(mem_new - t)/(1 - S(mem_new - t)) * i
        # Gradient of t: d_loss/d_t = (d_loss/d_o) * (d_o/d_x) * (d_x/d_t) = LS * S(mem_new - t)/(1 - S(mem_new - t)) * S(t) * -1
        # Gradient of mem_decay: d_loss/d_mem_decay = (d_loss/d_o) * (d_o/d_mem_decay) * (d_x/d_w) = LS * S(mem_new - t)/(1 - S(mem_new - t)) * mem_decay
        # LS for upstream layer: d_loss/d_i = (d_loss/d_o) * (d_o/d_x) * (d_x/d_i) = LS * S(mem_new - t)/(1 - S(mem_new - t)) * w

        # Then we just subtract the gradients from whatever we have, and that's an update to the parameters

        # +LS implies the neuron fired too much
        # -LS implies the neuron fired too little

        do_dx = torch.nn.functional.sigmoid(self.mem) * (
            1 - torch.nn.functional.sigmoid(self.mem)
        )

        dx_dt = -torch.nn.functional.sigmoid(self.threshold) * do_dx
        dx_dmem_decay = -self.mem_decay * do_dx
        # if LS is positive, the neuron fired too much, hence when - decay, subtracting should make it fire more sparsely, i.e. - = increase, hence negative

        ls = (do_dx * self.ls).unsqueeze(0)  # [1, out]
        ls = torch.broadcast_to(ls, (self.num_in, self.num_out))  # [in, out]

        in_trace = prev_layer.get_output_trace()

        input_tensor = in_trace.unsqueeze(-1)  # [in, 1]
        input_tensor = torch.broadcast_to(input_tensor, (self.num_in, self.num_out))

        dx_dw = ls * input_tensor  # d_ls/d_w = input
        di = ls * self.w  # d_ls/d_in = weight

        self.w -= dx_dw * self.lr
        self.t -= dx_dt * self.lr
        self.mem_decay -= dx_dmem_decay * self.lr
        upstream_ls = di.mean(dim=-1)

        return upstream_ls

    def forward(
        self,
        in_spikes: torch.Tensor,
        upstream_layer,
    ):
        if self.empty:
            return in_spikes

        upstream_ls = self.calculate_derivatives(upstream_layer)
        upstream_layer.update_ls(upstream_ls)

        raw_current = in_spikes @ self.w
        self.mem_old = self.mem
        self.mem *= torch.nn.functional.sigmoid(self.mem_decay)
        # mem_decay needs to pass through sigmoid as it must be between 0 and 1
        self.mem += raw_current

        softplus_thresh = torch.nn.functional.softplus(self.threshold)
        out_spikes = (self.mem >= softplus_thresh).float()
        self.mem -= out_spikes * softplus_thresh

        return out_spikes
