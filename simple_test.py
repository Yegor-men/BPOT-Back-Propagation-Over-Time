import torch
import architecture as arch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.set_grad_enabled(False)

num_elements = 100
random_tensor = torch.rand(num_elements).round()

bpot = arch.Network(
    num_inputs=num_elements,
    num_layers=3,
    num_outputs=num_elements,
    hidden_layer_size=num_elements,
    ls_decay=0,
    lr=1e-3,
)

errors = []
model_outputs = []

num_timesteps = 100
for i in range(num_timesteps):
    output, ls = bpot.forward(random_tensor, random_tensor)
    model_outputs.append(output)
    errors.append(ls.sum().item())

print(f"Training complete")

signals = bpot.get_learning_signals()
for index, signal in enumerate(signals):
    print(f"Signal for layer {index}: {signal[:5]}, loss={signal.sum()}")

output, ls = bpot.forward(random_tensor, random_tensor)
print(f"Sample inference")
print(f"\tReceived: {output.cpu()}")
print(f"\tExpected: {random_tensor.cpu()}")
print(f"\tL Signal: {ls.cpu()}")
print(f"\tLoss: {ls.sum().cpu()}")

import matplotlib.pyplot as plt
import numpy as np

# Plotting the average loss over time

ao = 1
average_errors = []

for index, error in enumerate(errors):
    if index < ao - 1:
        continue
    average_errors.append(sum(errors[index - ao : index]) / ao)

plt.plot(average_errors)
plt.xlabel("Time")
plt.ylabel("Loss")
plt.show()

# Plotting the model's output spike train

outputs_tensor = torch.stack(model_outputs)

spike_times = []

out_size = len(model_outputs[0])
num_timesteps = len(model_outputs)

for neuron_index in range(out_size):
    times = [t for t in range(num_timesteps) if outputs_tensor[t, neuron_index] == 1]
    spike_times.append(times)

plt.eventplot(spike_times, lineoffsets=range(out_size), linelengths=0.8, color="black")

plt.yticks(range(out_size), [f"Neuron {i}" for i in range(out_size)])
plt.xlabel("Timestep")
plt.ylabel("Neuron")
plt.title(f"Spike Train Visualization Over {num_timesteps:,} Timesteps")
plt.tight_layout()
plt.show()
