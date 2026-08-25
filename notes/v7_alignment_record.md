# V7 Local Alignment Record

## Status

- **Project:** Stability and Robustness of Signed and Directed GNNs Under Structural Perturbations
- **Document status:** Pre-results mentor-review draft
- **Manuscript version:** v7
- **Faculty mentor:** Dr. Yixuan He
- **Local working branch:** `work/v7-base-experiment`
- **Base commit:** `b0d6661edea4fb72d8f0781029509abd98266a1e`
- **Base commit date:** 2026-07-25
- **Remote publication status:** Not pushed
- **Mentor approval status:** Pending

## Purpose of This Branch

This local branch is used to align the July experimental repository with the v7 pre-results theory and methodology draft and to execute one documented MSGNN base experiment on ASU Sol.

This branch is not evidence that the theory, experimental protocol, or manuscript has been approved by Dr. He.

## Current Verified Evidence

The July repository has verified only:

- Python and package-environment availability;
- MSGNN, MagNet, and SSSNET imports;
- CPU constructor compatibility; and
- finite initialized parameters.

The repository has not yet verified:

- forward-pass correctness;
- task compatibility;
- training or convergence;
- clean-baseline validity;
- checkpoint reload reproducibility;
- structural perturbation utilities;
- fixed-checkpoint sensitivity;
- theory–implementation correspondence;
- multi-seed results; or
- manuscript-ready empirical evidence.

## V7 Base-Experiment Objective

The immediate objective is one small, reproducible MSGNN base pipeline on a mentor-confirmed signed-directed synthetic dataset, currently expected to be SDSBM.

The base experiment should verify:

1. environment and implementation identity;
2. dataset–task–model compatibility;
3. finite forward and backward passes;
4. one valid clean training run;
5. clean-baseline quality and class support;
6. deterministic checkpoint reload;
7. one small diagnostic structural perturbation, only after the clean gate passes;
8. traceable configurations, seeds, logs, manifests, outputs, and Sol resource use; and
9. a mentor-review evidence packet with appropriately limited claims.

## V7 Alignment Requirements

Before substantive execution, the local workflow must add or verify:

- fixed checkpoint, parameters, features, split, queries, labels, and evaluation code;
- separate data, split, training, perturbation, and evaluation seeds;
- the exact installed PyGSD version and relevant source identities;
- the executed MSGNN `K` convention;
- fixed charge and theorem-aligned constructor settings;
- representation and pre-softmax-logit extraction;
- deterministic evaluation with dropout disabled;
- perturbation eligibility and leakage rules;
- duplicate, self-loop, reverse-collision, and isolated-node handling;
- nested perturbation manifests;
- clean-baseline inclusion gates;
- theorem-assumption and conditional-coverage diagnostics; and
- progression from one base run to three, five, and preferably ten total independent clean-training runs, subject to mentor approval and feasibility.

## Claim Restrictions

This branch must not be used to claim:

- general MSGNN robustness;
- architecture rankings;
- statistical reliability from one run;
- full empirical validation of the theory;
- publication-level conclusions;
- mentor approval that has not been received; or
- equivalence between operator drift and task degradation.

## Local-Only Policy

Work may be committed locally for traceability. Nothing from this branch will be pushed to the public remote until the researcher deliberately authorizes it.

Because Sol scratch is temporary, local commits and important experiment artifacts must also receive a durable private backup outside scratch.

## Immediate Gate

Before training:

1. audit the existing code and active environment;
2. verify the MSGNN API and task;
3. create the base-experiment configuration and record;
4. run tiny forward, backward, checkpoint, and perturbation tests; and
5. stop if the implementation conflicts with v7.
