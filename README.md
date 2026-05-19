# MMAI 2026 - Emma Wang

Welcome to my site for Multimodal AI 2026.
This repo contains my homework assignments and random thoughts throughout the class.

## Bio
<img src="./imgs/profile.jpg" style="width:200px;">

Hi I'm Emma, a junior majoring in Computation and Cognition at MIT. I do research on engineering glutamate reponsive fMRI probes. Outside of class, I am a vasity sailor and enjoy eating, snowboarding, rock-climbing, and singing. 

## Final Project
For my final project, our group built **NeuroCLIP + LATA** — two diagnostic extensions on top of the [EEG2Video](https://nips.cc/virtual/2024/poster/95156) (NeurIPS 2024) benchmark for decoding visual concepts from EEG signals.

**NeuroCLIP** maps EEG features into a frozen CLIP concept space to ask which SEED-DV visual categories are actually decodable. Key finding: CLIP geometry doesn't explain EEG decodability — activity-rich concepts (sports, music, people) decode significantly better than passive scenes (Recall@1: 6.79% vs 3.83%, p = 5.95e-8).

**LATA (Latency-Aware Temporal Alignment)** learns a soft distribution over EEG-video delays during contrastive alignment to account for biological response latency. Key finding: all 20 SEED-DV subjects prefer a nonzero delay — 14/20 peak at ~790ms, and no subject peaks at zero lag.

[Group Repo](https://github.com/winstonqian/EEG2Video) | [My Copy](https://github.com/greenMangoes13/mmai/tree/master/EEGtoVideo)

## Homework
- [Homework 1 - Datasets](./homework/homework-1/) — Curated and analyzed a multimodal dataset, exploring preprocessing, annotation pipelines, and baseline evaluation strategies.
- [Homework 2 - Multimodal Fusion](./homework/homework-2/) — Designed and evaluated fusion architectures that combine multiple input modalities for a downstream classification task.
- [Homework 3 - Vision-Language Models](./homework/homework-3/) — Fine-tuned and probed a vision-language model on a custom task, analyzing cross-modal representations and zero-shot generalization.
- [Homework 4 - Reinforcement Learning for VLMs (GRPO)](./homework/homework-4/) — Implemented GRPO advantage computation and rule-based reward functions to RL-train Qwen3-VL-2B-Instruct on BUSI breast ultrasound classification; best run achieved 48.1% accuracy vs. 21.8% zero-shot baseline with 100% format compliance.
- [Homework 5 - Multimodal AI Agents](./homework/homework-5/) — Built a human-in-the-loop clinical decision-support agent for breast ultrasound triage using smolagents, with custom tools (BI-RADS reference, PubMed search, clinician review), adversarial benchmarking, safety evaluation, and Langfuse trace observability.


## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
