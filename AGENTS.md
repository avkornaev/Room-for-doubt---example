# AGENTS.md

Guidelines for AI agents, Codex, and student contributors working on the `Room for Doubt` repository.

This file defines research and implementation rules. It should guide code generation, experiment design, reporting, and repository maintenance.

---

# 1. Project Identity

`Room for Doubt` studies whether training dynamics can be distilled into inference-time uncertainty estimators.

The project is not just an engineering implementation. It is a research project. The code should support clear empirical claims, reproducible experiments, and honest negative results.

Core idea:

> Log simple per-sample training signals, compute trajectory descriptors offline, and test whether these descriptors can be predicted from final representations and used as uncertainty indicators.

The first version focuses on noisy-label uncertainty, especially CIFAR-10N.

Do not expand the project into broad OOD detection unless explicitly requested.

---

# 2. Methodological Structure

The project has three methodological stages.

## Stage 1: Training-Dynamics Logging and Post-Hoc Analysis

Train a classifier on noisy labels and log primitive per-sample signals.

The training loop should remain simple. Do not compute all descriptors inside training.

Log only primitive quantities sufficient for offline analysis:

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

## Stage 2: Descriptor Distillation

Freeze the trained encoder and train a lightweight model to predict trajectory descriptors from final representations.

Pipeline:

```text
image -> frozen encoder -> representation -> descriptor predictor -> predicted descriptors