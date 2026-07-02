# AGENTS.md

Guidelines for AI agents, Codex, and student contributors working on the `Room for Doubt` repository.

This file defines research and implementation rules. It should guide code generation, experiment design, reporting, and repository maintenance.

---

## 1. Project Identity

`Room for Doubt` studies whether training dynamics can be used as a source of uncertainty information.

The project is not only an engineering task. It is a research project. The code should support clear empirical claims, reproducible experiments, compact analysis, and honest negative results.

Core idea:

> Log primitive per-sample training signals, compute trajectory descriptors offline, and test whether training-dynamics information can be distilled into inference-time uncertainty estimators.

The first version focuses on noisy-label uncertainty, especially CIFAR-10N and optionally CIFAR-100N.

Do not expand the project into broad OOD detection unless explicitly requested.

---

## 2. Methodological Structure

The project has three methodological stages.

### Stage 1: Training-Dynamics Logging and Post-Hoc Analysis

Train a classifier on noisy labels and log primitive per-sample signals.

The training loop should remain simple. Do not compute all possible descriptors inside training.

Minimal logged signals:

- sample ID;
- epoch;
- noisy label;
- clean label for evaluation only;
- per-sample loss;
- predicted class probabilities.

From these logs, compute descriptors offline:

- assigned-label confidence;
- maximum softmax probability;
- entropy;
- margin;
- Area Under the Margin;
- confidence variability;
- forgetting count;
- mean loss;
- final loss;
- final confidence;
- first learned epoch;
- other reasonable descriptors.

### Stage 2: Descriptor Distillation

Freeze the trained encoder and train a lightweight model to predict trajectory-derived information from final representations.

Possible targets:

- hand-crafted descriptor vector `s_i`;
- learned trajectory representation `r_i`;
- both descriptor and learned trajectory representations.

The first baseline should be simple and interpretable:

```text
image -> frozen encoder -> representation -> descriptor predictor -> predicted descriptors