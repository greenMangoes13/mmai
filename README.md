# MMAI 2026 - Emma Wang

Welcome to my site for Multimodal AI 2026.
This repo contains my homework assignments and random thoughts throughout the class.

## Bio
<img src="./imgs/profile.jpg" style="width:200px;">

Hi I'm Emma, a junior majoring in Computation and Cognition at MIT. I do research on engineering glutamate reponsive fMRI probes. Outside of class, I am a vasity sailor and enjoy eating, snowboarding, rock-climbing, and singing. 

## Final Project
For my final project, our group audited and extended [EEG2Video](https://nips.cc/virtual/2024/poster/95156) — a NeurIPS 2024 model that reconstructs video from EEG brain signals. My contributions include:

- Benchmarked the EEG-VP semantic classification pipeline (DE/PSD features, MLP architecture) and established a ~4.15% Top-1 baseline on 40 video categories
- Extended the codebase to natively handle raw 200Hz EEG (`T=400`) by dynamically scaling CNN spatial dimensions
- Audited for data leakage in train/test normalization — found a **13x variance drop** in PSD features when leakage was removed
- Ran clip-index stratification to confirm the model decodes stimulus perception, not protocol artifacts
- Prototyped temporal attention pooling (`O(T²)`); found it too compute-expensive vs. CNN pooling

[Group Repo](https://github.com/winstonqian/EEG2Video) | [My Copy](https://github.com/greenMangoes13/mmai/tree/master/EEGtoVideo)

## Homework
- [Homework 1 - Datasets](./homework/homework-1/) — Curated and analyzed a multimodal dataset, exploring preprocessing, annotation pipelines, and baseline evaluation strategies.
- [Homework 2 - Multimodal Fusion](./homework/homework-2/) — Designed and evaluated fusion architectures that combine multiple input modalities for a downstream classification task.
- [Homework 3 - Vision-Language Models](./homework/homework-3/) — Fine-tuned and probed a vision-language model on a custom task, analyzing cross-modal representations and zero-shot generalization.
- [Homework 4 - Reinforcement Learning for VLMs (GRPO)](./homework/homework-4/) — Implemented GRPO advantage computation and rule-based reward functions to RL-train Qwen3-VL-2B-Instruct on BUSI breast ultrasound classification; best run achieved 48.1% accuracy vs. 21.8% zero-shot baseline with 100% format compliance.
- [Homework 5 - Multimodal AI Agents](./homework/homework-5/) — Built a human-in-the-loop clinical decision-support agent for breast ultrasound triage using smolagents, with custom tools (BI-RADS reference, PubMed search, clinician review), adversarial benchmarking, safety evaluation, and Langfuse trace observability.


## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
