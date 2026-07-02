# Room for Doubt

**Distilling training dynamics for inference-time uncertainty estimation**

## Overview

Standard neural networks often produce confident predictions even for mislabeled, ambiguous, or unstable samples. Classical uncertainty scores such as maximum softmax probability, entropy, or energy are computed only from the final model output. They ignore how each sample behaved during training.

This project studies a different hypothesis:

> A model's doubt may be partially encoded in the temporal trajectory of learning.

Instead of asking only *what does the model predict now?*, we also ask:

* was this sample learned early or late?
* was it forgotten during training?
* was its assigned-label confidence stable?
* did its margin remain positive or oscillate?
* can this historical behavior be predicted at inference time?

The project aims to convert training dynamics into useful uncertainty indicators for noisy-label detection, selective prediction, and uncertainty-aware inference.

---

## Main Hypothesis

Let

[
\mathcal{D} = {(x_i, \tilde{y}*i)}*{i=1}^{N}
]

be a dataset with possibly noisy labels, where (x_i) is an input image and (\tilde{y}_i) is the observed label.

During training, each sample produces a temporal trajectory:

[
\tau_i = {q_i^{(t)}}_{t=1}^{T}.
]

In this project, the trajectory is intentionally represented by simple primitive signals, such as per-sample loss and predicted class probabilities. More complex descriptors are computed later, outside the training loop.

A trajectory can be compressed into a descriptor vector:

[
s_i = A(\tau_i),
]

for example:

[
s_i =
[
\mathrm{AUM}_i,
\mathrm{MeanConf}_i,
\mathrm{VarConf}_i,
\mathrm{Forgetting}_i
].
]

The central question is whether a final frozen representation

[
h_i = f_\theta(x_i)
]

contains enough information to predict this historical descriptor:

[
g_\phi(h_i) \approx s_i.
]

If this is possible, then training dynamics can be distilled into an inference-time uncertainty estimator.

---

## Research Questions

1. Can training-dynamics descriptors identify mislabeled or unstable samples better than static confidence scores?

2. Can historical training-dynamics descriptors be predicted from final representations?

3. Can predicted descriptors serve as useful uncertainty indicators at inference time?

4. Can these uncertainty indicators improve selective prediction or uncertainty-weighted prediction?

5. Does conditional flow matching provide useful probabilistic information beyond deterministic descriptor regression?

---

## Project Scope

The first version of the project focuses on **noisy-label uncertainty**.

The project does **not** focus on general OOD detection in the initial version.

Primary goals:

* train a classifier on noisy labels;
* log simple per-sample training trajectories;
* compute trajectory descriptors offline;
* evaluate post-hoc dynamics as uncertainty indicators;
* distill these descriptors into an inference-time predictor;
* evaluate predicted uncertainty against static baselines;
* optionally use predicted uncertainty with weak TTA or dropout;
* explore conditional flow matching as an advanced extension.

The expected result is a reproducible empirical study suitable for a short student research paper.

---

## Dataset

The primary dataset is **CIFAR-10N**, which provides human-annotated noisy labels for CIFAR-10 images.

Recommended splits:

* `aggregate`;
* `worse`;
* optionally other CIFAR-10N noisy-label variants.

Clean CIFAR-10 labels may be used for evaluation, but not for training the base classifier or constructing uncertainty targets.

---

## Methodological Stages

The project has three methodological stages.

---

## Stage 1: Training-Dynamics Logging and Post-Hoc Analysis

Train a standard classifier on noisy labels.

During training, do not compute every possible metric inside the training loop. Instead, log simple primitive signals that are sufficient to reconstruct many descriptors later.

Minimal per-sample logging:

[
q_i^{(t)} =
[
\ell_i^{(t)},
p_i^{(t)}
],
]

where:

* (\ell_i^{(t)}) is the loss for sample (x_i) at epoch (t);
* (p_i^{(t)} \in \mathbb{R}^{C}) is the predicted probability vector over classes.

The log should preserve:

* sample ID;
* epoch;
* noisy label;
* clean label for evaluation only;
* per-sample loss;
* predicted class probabilities.

From these primitive logs, descriptors can be computed offline.

Examples:

### Assigned-label confidence

[
\mathrm{Conf}_i^{(t)}
=====================

p_i^{(t)}(\tilde{y}_i).
]

### Maximum softmax probability

[
\mathrm{MSP}_i^{(t)}
====================

\max_k p_i^{(t)}(k).
]

### Assigned-label margin

If full probabilities are stored, a log-probability margin can be computed as:

[
m_i^{(t)}
=========

## \log(p_i^{(t)}(\tilde{y}_i) + \epsilon)

\max_{k \neq \tilde{y}_i}
\log(p_i^{(t)}(k) + \epsilon).
]

This is equivalent to the corresponding logit margin up to numerical precision.

### Area Under the Margin

[
\mathrm{AUM}_i
==============

\frac{1}{T}
\sum_{t=1}^{T}
m_i^{(t)}.
]

### Confidence variability

[
\mathrm{VarConf}_i
==================

\mathrm{Std}_{t}
[
p_i^{(t)}(\tilde{y}_i)
].
]

### Forgetting count

Let

[
c_i^{(t)}
=========

\mathbb{1}
[
\arg\max_k p_i^{(t)}(k) = \tilde{y}_i
].
]

Then

[
\mathrm{Forgetting}_i
=====================

\sum_{t=1}^{T-1}
\mathbb{1}
[
c_i^{(t)} = 1
\land
c_i^{(t+1)} = 0
].
]

This design keeps the training loop simple and makes descriptor design flexible.

Students can add new descriptors later without retraining the classifier, as long as the required primitive signals were logged.

Possible post-hoc descriptors:

* Area Under the Margin;
* mean assigned-label confidence;
* confidence variability;
* forgetting count;
* mean loss;
* final loss;
* final confidence;
* final margin;
* first learned epoch;
* last forgotten epoch;
* entropy statistics across epochs.

Possible static baselines:

* maximum softmax probability;
* entropy;
* final loss;
* final margin;
* final assigned-label confidence.

The main evaluation target is noisy-label detection and uncertainty ranking.

---

## Stage 2: Descriptor Distillation for Inference-Time Uncertainty

After Stage 1, each sample has a descriptor vector:

[
s_i = A(\tau_i).
]

Then freeze the trained encoder and extract final representations:

[
h_i = f_\theta(x_i).
]

Train a lightweight descriptor predictor:

[
g_\phi(h_i) = \hat{s}_i.
]

The goal is to predict post-hoc training-dynamics descriptors without access to the full training trajectory.

This tests whether historical learning behavior is encoded in the final representation space.

The predicted descriptor vector can be converted into an uncertainty score:

[
u_i = U(\hat{s}_i).
]

The score may be:

* hand-designed;
* calibrated on a validation set;
* learned by a shallow model;
* optimized for ranking suspicious samples;
* optimized for selective prediction.

A simple first version may combine normalized predicted descriptors:

[
u_i =
\alpha_1(-\widehat{\mathrm{AUM}}_i)
+
\alpha_2\widehat{\mathrm{VarConf}}_i
+
\alpha_3\widehat{\mathrm{Forgetting}}_i
+
\alpha_4(1-\widehat{\mathrm{MeanConf}}_i).
]

This formula is not fixed. It is only a reasonable starting point.

A possible additional experiment is uncertainty-aware prediction aggregation.

For weak test-time augmentations or dropout samples,

[
p_1, p_2, ..., p_K
]

can be combined using uncertainty-dependent weights:

[
\bar{p}(y \mid x)
=================

\frac{
\sum_{k=1}^{K} w_k p_k(y \mid x)
}{
\sum_{k=1}^{K} w_k
},
]

where

[
w_k = \exp(-\tau u_k)
]

or another decreasing function of predicted uncertainty.

The question is whether uncertain augmented predictions should contribute less to the final decision.

---

## Stage 3: Conditional Flow Matching Extension

The deterministic descriptor predictor estimates one vector:

[
g_\phi(h_i) = \hat{s}_i.
]

The advanced extension studies a probabilistic alternative:

[
p_\psi(s_i \mid h_i).
]

A conditional flow-matching model can generate multiple plausible descriptor vectors for the same representation:

[
\hat{s}_i^{(1)}, \hat{s}_i^{(2)}, ..., \hat{s}_i^{(M)}.
]

The variance or disagreement among generated descriptors may provide an additional uncertainty signal.

This stage is exploratory. It should not block the main deterministic pipeline.

The main question is not whether flow matching is fashionable, but whether it adds measurable value beyond deterministic descriptor regression.

---

## Compact Experimental Outputs

Experiment outputs should be designed for later analysis with ChatGPT and for writing the Results section of the paper.

Each important experiment should generate:

* compact result tables;
* plots;
* saved configuration;
* trained checkpoints when relevant;
* `report.md`.

The compact tables should save tokens during later analysis.

Guidelines for compact tables:

* include only essential columns;
* use short method names;
* round metrics to 3-4 meaningful digits;
* keep detailed raw artifacts separately;
* report mean and standard deviation for multi-seed experiments;
* avoid unnecessary per-sample rows in summaries.

Example compact table:

```csv
method,split,seed,auroc,auprc,aurc,ece,acc
msp,worse,0,0.712,0.481,0.184,0.092,0.821
entropy,worse,0,0.728,0.496,0.177,0.087,0.821
aum,worse,0,0.803,0.586,0.142,0.071,0.821
pred_desc,worse,0,0.819,0.604,0.136,0.068,0.821
```

Raw trajectory logs may be stored with higher precision if needed, but reports and compact tables should use rounded values.

The goal is not to preserve excessive numerical detail. The goal is to preserve enough information for interpretation, comparison, and paper writing.

---

## Experiment Reports

The `report.md` file is a mandatory artifact for meaningful experiments.

It should explain:

* what was tested;
* why the experiment was needed;
* which configuration was used;
* which dataset split was used;
* what the main results were;
* what failed;
* what can be claimed in the paper;
* what should be tested next.

These reports are expected to be used later for ChatGPT-assisted analysis and for writing the Results section.

A good report is more valuable than a large unstructured log.

---

## Implementation Preferences

The project should use a modern research-code stack:

* PyTorch;
* PyTorch Lightning;
* Hydra;
* local experiment outputs;
* Markdown reports;
* compact CSV tables;
* plots for analysis and paper figures.

No external experiment tracker is required in the first version.

The implementation should remain flexible enough for students to make reasonable design decisions and compare alternatives.

Detailed repository structure, exact scripts, config hierarchy, and implementation rules should be specified in `AGENTS.md` or planned by Codex, not overdefined in this README.

---

## Minimal Success Criteria

The project is successful if it produces:

1. a working noisy-label training pipeline;
2. primitive per-sample training logs;
3. offline-computed trajectory descriptors;
4. post-hoc uncertainty baselines;
5. a descriptor-distillation model;
6. inference-time uncertainty scores;
7. reproducible reports and compact result tables;
8. paper-ready figures and experimental summaries.

The conditional flow-matching model is an advanced extension. It is valuable if time allows, but the core contribution should remain clear even without it.

---

## Expected Final Outcome

The final repository should support the research claim:

> Training dynamics contain useful information about uncertainty, and part of this information can be distilled into inference-time predictors.

The final paper should honestly report both positive and negative results.

Important possible outcomes:

* training-dynamics descriptors outperform static confidence scores;
* descriptor distillation works only for some descriptors;
* predicted descriptors improve noisy-label ranking but not calibration;
* uncertainty-weighted TTA/dropout helps only in limited regimes;
* flow matching adds useful variance information;
* flow matching does not justify its complexity.

Negative results are acceptable if they are well documented and scientifically interpretable.
