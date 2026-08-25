# Masters-Thesis
## Determining the Sufficient Number of Shots for Noisy Quantum Devices
### Predicting the Hellinger convergence of measurements using shallow Shot probes and Circuit features

On noisy intermediate-scale quantum (NISQ) devices [1], every quantity of interest
must be reconstructed statistically from repeated measurements, or shots, making the
shot count the fundamental unit of computational work and a scarce, costly resource.
That cost is concrete rather than abstract. Access to quantum hardware is overwhelmingly
cloud-based and billed by usage, with providers charging per shot or per submitted task,
so a measurement budget translates directly into money. It also translates into time:
every shot consumes a full cycle of state preparation, gate execution, measurement and
reset.

The efficient use of this budget is complicated by a two-sided trade-off. Estimating
a quantum state from 𝑁 independent measurement outcomes is a sampling problem.
Attaining a precision(in other words reducing the error) 𝜀 costs 𝑁 ∼ 𝜎2/𝜀2 shots
where 𝜎 is the standard deviation of the single-shot outcomes. This inverse-quadratic
scaling is severe enough to have been identified as a principal obstacle to practical
quantum advantage in variational algorithms [2, 3]. The other factor that keeps us
from estimating the ideal quantum state is that the samples are drawn from the device’s
noise-corrupted distribution rather than the ideal one, the measured Hellinger distance
between them decays towards an irreducible noise floor that reflects systematic bias and
that no additional sampling can lower, since reducing such bias requires error-mitigation
techniques rather than more shots [4]. Beyond the point at which this floor is reached,
further measurement is wasted. Too few shots, conversely, leave estimates so noisy that
the optimization they support becomes unreliable, a concern that has motivated dedicated
adaptive shot optimizers [5]. The quantity that should be sought is therefore not the
smallest but the optimal number of shots—the saturation point at which the decay has
3
effectively flattened. Determining this point by direct experimentation is self-defeating,
since probing how many shots a circuit requires itself consumes shots.

This thesis addresses the problem of estimating that saturation point cheaply, ahead of
execution. The approach rests on two hypotheses. The first is that the fluctuations of
the Hamming distance, specifically the standard deviation of the per-shot Hamming distance
measured on mirrored version of the circuit, whose ideal output is a single known
basis state obtained by construction rather than by classical simulation is related to the
Hellinger decay of the halved(original) version of the circuit, and so provide an inexpensive,
simulation-free proxy for it. The second is that a circuit’s static structural features,
captured by the hardware-agnostic SupermarQ feature vector, are related to how quickly
its output degrades under noise, because those features describe precisely the connectivity,
entanglement content and depth through which error accumulates. Together these
suggest that a machine learning model can predict a circuit’s full decay behaviour from
its structure combined with a small amount of cheaply obtained measurement.

The methodology proceeds in stages. A structurally diverse collection of variational
circuits spanning two ansatz families, three entanglement topologies and a range of
depths is generated, and mirrored counterparts are constructed by composing each circuit
with its inverse. These are simulated under a depolarizing noise model across a sweep of
shot counts, yielding Hellinger and Hamming decay curves. Each curve is smoothed and
fitted to an exponential decay model, compressing it into three parameters from which
the saturation point is extracted as the shot count at which the fitted slope falls below a
chosen threshold. A compact neural network is then trained by supervised regression,
using a Huber loss for robustness to the fit-derived targets, to map a circuit’s structural
features together with two shallow-shot Hamming measurements onto these three decay
parameters. To test generalization honestly, the held-out circuits are constructed with
4 layer counts, and therefore structures and feature vectors, that are absent from the training
data.

The trained model reproduces the saturation points of unseen circuits accurately. Evaluated
at the most demanding, plateau-onset threshold, the predicted shot budgets deviate
from the true ones by a mean absolute error of roughly one to one-and-a-half increments
of the shot sweep (each increment is 20 shots), corresponding to a mean absolute
percentage error of about three percent, with the large majority of predictions falling
within two increments. Averaging over a small range of nearby near-plateau thresholds
makes the estimate even more robust, tightening the error distribution and eliminating
extreme outliers. The results support both hypotheses and indicate that a near-optimal
shot budget can indeed be anticipated from a circuit’s structure together with a small
amount of inexpensive measurement, rather than discovered through costly experimentation.
A modest systematic tendency to underestimate the saturation point is identified
as the principal limitation, along with the reliance on a single simulated noise model,
and directions for addressing both are discussed.

