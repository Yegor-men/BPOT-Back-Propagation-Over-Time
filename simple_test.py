import torch
import architecture as arch


bpot = arch.Network(
    num_inputs=5,
    num_layers=3,
    num_outputs=5,
    hidden_layer_size=5,
)

# just 5, 5, 5, the idea is that [1, 0, 1, 0, 1] should return itself

test_tensor = torch.tensor([1, 0, 1, 0, 1])

errors = []

num_timesteps = 100000
for i in range(num_timesteps):
    output, ls = bpot.forward(test_tensor, test_tensor)
    errors.append(ls.sum().item())

print(f"Training complete")

signals = bpot.get_learning_signals()
for index, signal in enumerate(signals):
    print(f"Signal for layer {index}: {signal[:5]}, loss={signal.sum()}")

output, ls = bpot.forward(test_tensor, test_tensor)
print(f"\tSample inference - output: {output}, learning signal:  {ls}, error: {ls.sum()}")

ao = 100
average_errors = []

for index, error in enumerate(errors):
    if index < ao - 1:
        continue
    average_errors.append(sum(errors[index - ao : index]) / ao)

import matplotlib.pyplot as plt

plt.plot(average_errors)
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()
