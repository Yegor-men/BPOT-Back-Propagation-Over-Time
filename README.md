# BPOT: Back Propagation Over Time

BPTT (Back Propagation Through Time) is widely and rightfully considered to be biologically implausible. It imples not
only storing raw inputs over an arbitrary amount of time, it also implies some central control that can pause learning
and all this in one timestep.

However, the "calculations" for backpropagation at any given layer is incredibly simple, largely retrieving values than
calculating anything. Perhaps on a local scale backpropagation exists?

The concept of BPOT is hence to have backpropagation entirely locally, hence it's the exact same backpropagation,
instead it happens over time, rather than through time. It implies that a neuron is able to send a signal upstream, but
frankly that is much mroe reasonable than BPTT. Point is BPOT is actually biologically viable.

The goal of this "library" (not sure what of it for now) is a kind of expansion upon pytorch and snntorch. It's unusual
because it heavily relies on the concept of single lifetime learning (SLL) (I couldn't think of a better name), and
hence has no batches. It's designed to be trained in one lifetime, relying on traces and decomposing said traces to find
the "average" inputs and outputs.

Functionally, it's the same as normal backpropagaion, nothing new here, but for the sake of comfort and the nonstandard
biological restrictions, importing bpot disables pytorch autograd. Doing things like .backward is a little more manual,
although ideally I'd like to figure it out how to make it cleaner and more comfortable.
