# arXiv v2 replacement metadata for arXiv:2604.23743

This file collects the fields to enter in the arXiv "replace" web form when
uploading arxiv_v2.tar.gz as version 2 of arXiv:2604.23743. arXiv permits a
title change on replacement, so the title below supersedes the v1 title.

## Title (new)

An architectural capacity ceiling, not a barren plateau: why a fixed-encoding
variational quantum circuit cannot fit the Lorenz-63 attractor

## Authors

Tushar Pandey (Texas A&M University)

## Abstract (plain text for the arXiv abstract field)

Variational quantum circuits train poorly on chaotic forecasting, a failure
usually blamed on barren plateaus (exponentially vanishing gradients). Using an
exactly simulable four-qubit variational quantum physics-informed circuit fit to
the Lorenz-63 system, we show the barren-plateau explanation fails and identify
the failure as an architectural capacity ceiling fixed by the circuit's
time-encoding, not its trainable depth. Four measurements support this. (i) A
McClean-comparable gradient-variance estimator sits at the local-cost
Haar/2-design scale 2^(-2n) = 3.9e-3 at n = 4; on its structurally live
parameters it decays about ninefold with depth, then saturates there, large
enough to train, not an exponential collapse to zero. (ii) At a common nominal
budget of 200 optimiser iterations (600, in three stages, for layer-wise), three
optimiser families (gradient descent, layer-wise, SPSA) reach the same order of
magnitude of loss, with finite-difference and exact parameter-shift gradients
agreeing to about 2e-5 relative as a consistency check, so no optimiser unlocks a
better basin. (iii) The output-Jacobian rank saturates at 33 from five layers on,
so depth buys no new output directions. (iv) A Fourier analysis explains why: the
qubit-1 phase encoding acts on the initial |0> and is inert, so the maximum
accessible frequency is 2.5/t_max = 0.83 Hz, identical at every depth and a factor
of about 4.4 below the narrowest Lorenz component's bandwidth. The corrected band
has dimension 1 + 2x5 = 11 per observable, and 3x11 = 33 equals the measured rank
ceiling exactly, unifying the two diagnostics. A trained depth sweep agrees: mean
loss improves with depth, then flattens once the rank saturates. We correct our
earlier preprint diagnosis, which compared unnormalised gradient norms to the
McClean threshold, and place the advantage of fixed reservoirs and classical
echo-state networks in architecture, not quantum mechanics.

## Comments field (suggested)

v2: major revision; corrected normalised gradient-variance analysis (McClean-
comparable local-cost estimator against the same-ansatz 2-design floor, split
into structurally live and dead parameters), added optimiser-independence and
output-Jacobian capacity diagnostics, and a corrected Fourier-expressivity
ceiling; reframed the reservoir/ESN advantage as architectural rather than
quantum; title changed. 17 pages, 3 figures. Prepared with IOP iopart class
(submitted to Machine Learning: Science and Technology).

## License recommendation

Keep the existing v1 license. arXiv does not allow the license to be made more
restrictive on replacement, and reusing the v1 license avoids any licensing
conflict on the replacement. Select the same license that was chosen for v1 of
arXiv:2604.23743 (do not switch to a different or more restrictive license).

## Notes

- Vendored class files (iopart.cls, iopams.sty, iopart10.clo, iopart12.clo) are
  the genuine IOP Publishing files (LPPL, "Current Maintainer: IOP Publishing
  Ltd"), required because arXiv's TeX Live does not ship iopart. Same approach as
  the team's earlier llncs upload.
