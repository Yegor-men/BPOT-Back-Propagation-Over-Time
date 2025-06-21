import torch


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.set_grad_enabled(False)

import complex_architecture as arch


class GarbageNetwork:
    def __init__(
        self,
    ) -> None:
        self.layers = []

        for i in range(3):
            self.layers.append(
                arch.Layer(
                    num_in=10,
                    num_out=10,
                )
            )

        self.garbage = arch.Layer(
            num_in=10,
            num_out=10,
            empty_layer=True,
        )

    def get_ls(
        self,
    ):
        for layer in self.layers:
            print(layer.ls)

    def forward(
        self,
        in_spikes,
        expected_output,
    ):
        in_spikes = in_spikes.to(self.layers[0].w)
        expected_output = expected_output.to(self.layers[0].w)
        
        o, ls = self.garbage.forward(in_spikes, self.garbage)
        o, ls = self.layers[0].forward(o, self.garbage)
        o, ls = self.layers[1].forward(o, self.layers[0])
        o, ls = self.layers[2].forward(o, self.layers[1])
        
        loss = expected_output - o
        self.layers[2].update_ls(loss)
        
        print(f"Loss: {(loss**2).sum()}, {loss}")
        
        return o


net = GarbageNetwork()
garbage = torch.rand(10).round()

init_w = net.layers[0].w.clone()

for i in range(1000):
    out = net.forward(garbage, garbage)

fin_w = net.layers[0].w.clone()

print((fin_w - init_w).sum())

net.get_ls()