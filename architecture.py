import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.set_grad_enabled(False)


class Layer:
    def __init__(
        self,
        num_in: int,
        num_out: int,
        threshold: float = 1,
        mem_decay: float = 0.99,
        ls_decay: float = 0,
        lr: float = 1e-6,
        device: str = "cuda",
    ) -> None:
        self.num_in = num_in
        self.num_out = num_out

        self.mem_decay = mem_decay
        self.ls_decay = ls_decay
        self.lr = lr

        self.w = torch.randn(num_in, num_out).to(device)
        self.mem = torch.zeros(num_out).to(self.w)
        self.threshold = torch.ones(num_out).to(self.w) * threshold

        self.ls = torch.zeros_like(self.mem)

        self.in_trace = torch.zeros(num_in).to(self.w)

    def update_ls(
        self,
        learning_signal: torch.Tensor,
    ) -> None:
        assert learning_signal.shape == (
            self.num_out,
        ), f"Learning Signal update: expected size ({self.num_out},), got {learning_signal.shape} instead"
        learning_signal = learning_signal.to(self.ls)

        self.ls *= self.ls_decay
        self.ls += learning_signal

    def backward(
        self,
    ) -> torch.Tensor:
        ls = self.ls.unsqueeze(0)  # [1, out]
        ls = torch.broadcast_to(ls, (self.num_in, self.num_out))  # [in, out]

        input_tensor = self.in_trace.unsqueeze(-1)  # [in, 1]
        input_tensor = torch.broadcast_to(
            input_tensor, (self.num_in, self.num_out)
        )  # [in, out]

        delta_w = ls * input_tensor  # d_ls/d_w = input
        delta_input = ls * self.w  # d_ls/d_in = weight

        self.w -= delta_w * self.lr  # goes down the gradient

        passed_ls = delta_input.mean(dim=-1)

        return passed_ls

    def forward(
        self,
        in_spikes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert in_spikes.shape == (
            self.num_in,
        ), f"Forward: expected size ({self.num_in},), got {in_spikes.shape} instead"
        in_spikes = in_spikes.to(self.w)

        back_pass_ls = self.backward()

        self.in_trace *= self.mem_decay
        self.in_trace += in_spikes

        current = in_spikes @ self.w
        self.mem *= self.mem_decay
        self.mem += current

        out_spikes = (self.mem >= self.threshold).float()

        self.mem -= out_spikes * self.threshold

        return out_spikes, back_pass_ls


class Network:
    def __init__(
        self,
        num_inputs: int,
        num_layers: int,
        num_outputs: int,
        hidden_layer_size: int,
        ls_decay: float = 0,
        lr: float = 1e-3,
    ) -> None:
        self.layers = [
            Layer(
                num_in=hidden_layer_size,
                num_out=hidden_layer_size,
                ls_decay=ls_decay,
                lr=lr,
            )
            for _ in range(num_layers)
        ]
        self.layers[0] = Layer(
            num_in=num_inputs,
            num_out=hidden_layer_size,
            ls_decay=ls_decay,
            lr=lr,
        )
        self.layers[-1] = Layer(
            num_in=hidden_layer_size,
            num_out=num_outputs,
            ls_decay=ls_decay,
            lr=lr,
        )

        for index, layer in enumerate(self.layers):
            print(f"Layer {index} - in: {layer.num_in}, out: {layer.num_out}")

    def forward(
        self,
        spike_input: torch.Tensor,
        expected_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spike_input = spike_input.to(self.layers[0].w)
        expected_output = expected_output.to(self.layers[0].w)

        o, ls = self.layers[0].forward(spike_input)
        # forward pass through the first layer, but no need to pass back learning signal to the input layer

        for i in range(1, len(self.layers)):
            o, ls = self.layers[i].forward(o)
            # get the output and learning signal to pass back
            self.layers[i - 1].update_ls(ls)
            # pass back the learning signal back one layer to the previous one

        ls = (
            o - expected_output
        )  # the gradient of the slope, NOT the direction which the neuron has to go
        # need a custom learning signal for the final layer
        self.layers[-1].update_ls(ls)
        # update the last layer (it's outside the loop)

        return o, ls

    def get_learning_signals(
        self,
    ) -> list[torch.Tensor]:
        ls = []
        for layer in self.layers:
            ls.append(layer.ls)
        return ls
