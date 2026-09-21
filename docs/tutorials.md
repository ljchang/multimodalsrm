# Tutorials planned for the model handoff

Runnable model tutorials will be transferred with the verified implementation. The current package does not expose either estimator. These requirements define the examples to add; they are not executable examples of the scaffold.

## R-MSRM tutorial

Use synthetic native-time observations with two modalities and multiple participants/runs. Fit an exact common response, inspect shared parametric kernels and fit diagnostics, infer an independent held-out run, predict a target with its observations excluded, and calibrate a new participant. Make training-only preprocessing and response-support masks explicit. Show `estimate=True` with `fixed={...}` for selecting estimated response parameters. Report prediction and response recovery separately.

## GP-MSRM tutorial

Use the same named data convention and explain optional Bayesian installation, fixed versus learned quantities and CPU float64. Demonstrate MAP and the posterior capabilities verified by the final handoff, including supported updates and archive roundtrips. Display convergence diagnostics before interpreting intervals. Distinguish marginal query uncertainty from joint trajectory uncertainty and document the actual backend/response-family matrix.

Both tutorials must run from the installed distribution, use only small synthetic data, declare random seeds and avoid introducing participant graphs, private trajectories or learned FIR responses into the initial release.
