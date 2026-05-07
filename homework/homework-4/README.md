# Homework 4 - Reinforcement Learning for VLMs (GRPO)

**Notebook:** [Google Colab](ADD_YOUR_COLAB_LINK_HERE)
**Data:** [Google Drive](https://drive.google.com/drive/folders/1a3FV1p9EALz_J_TzwrJCDlShxR_tSqbg?usp=drive_link)

## Overview

HW4 was about **Group Relative Policy Optimization (GRPO)** — a reinforcement learning algorithm for aligning language models with reward signals, without needing a learned critic model. The setup reused the BUSI breast ultrasound dataset from HW3 and the same base model, **Qwen3-VL-2B-Instruct**, but replaced supervised fine-tuning with RL training using rule-based reward functions.

## Reading: From PPO to GRPO

The reading covered the DeepSeekMath paper and the Illustrated GRPO guide. A few threads stood out:

**On GRPO vs. PPO**: Standard RLHF methods like PPO require four models — policy, reference, reward model, and a critic (value model) trained in parallel. GRPO eliminates the critic entirely by estimating the baseline from the group of completions generated for the same prompt. For each question, the model samples G outputs and receives a scalar reward for each. Advantages are computed as:

```
A_i = (r_i - mean(r)) / (std(r) + ε)
```

where the mean and std are taken over the G outputs in the group. Outputs above average get positive advantage and are reinforced; below-average outputs are suppressed. When all G completions receive the same reward, every advantage is zero and the model gets no gradient — which makes sense, since if all outputs are equally good or bad there's nothing to learn from the comparison. The main tradeoff is that GRPO requires generating G outputs per prompt at every training step, which is expensive at inference time. The DeepSeekMath results showed GRPO improved performance on the MATH benchmark from 46.8% to 51.7% using this approach.

**On reward design**: Learned reward models can generalize to subjective tasks, but are susceptible to reward hacking — the policy can learn to exploit model biases rather than produce genuinely better outputs. DeepSeekMath notes that even carefully annotated datasets like PRM800K contain ~20% incorrect labels, which undermines the learned signal as the policy improves. Rule-based rewards (checking if an answer matches ground truth, or running generated code in a sandbox) are more robust because they are grounded in objective criteria the model cannot game through learned biases. The risk of reward hacking with rule-based rewards is lower for the verified dimension, though the model can still learn to superficially match format without meaningful reasoning.

**On SFT vs. GRPO**: SFT trains the model to maximize the probability of the ground-truth answer token by token — every token contributes equally to the gradient. GRPO learns from the model's own generations: it samples outputs, receives scalar rewards, and updates based on relative comparisons within the group. SFT is better when labeled data is available and the task has clear correct answers. GRPO is better when the model already has latent ability and needs its output distribution calibrated — as the DeepSeekMath analysis shows, GRPO primarily improves Maj@K (majority vote robustness) rather than Pass@K (raw capability).

## Implementation: Advantage Computation and Reward Functions

The core of the homework was implementing the GRPO advantage computation from scratch. The `compute_grpo_advantage()` function groups rewards by prompt, normalizes within each group, and returns per-token advantages. All 5 unit tests passed.

Two rule-based reward functions were implemented:

- **Accuracy reward** (binary): 1.0 if the extracted answer matches ground truth, 0.0 otherwise. Uses label normalization to handle common model paraphrases ("fibroadenoma" → benign, "breast cancer" → malignant).
- **Format reward** (tiered): 1.0 if the response uses "Answer:" followed by a valid label, 0.5 if it uses "Answer:" with an unrecognized label, 0.0 if no "Answer:" marker at all. This prevents the reward from being zero everywhere early in training, giving the model a learning signal even before it gets answers right.

The training dataset was built from 624 BUSI training images with an instruction suffix asking the model to think step-by-step and output its final label after "Answer:".

## Training: Hyperparameter Sweep

Three GRPO configurations were trained for 100 steps each:

| Run | LR | Temp | LoRA r/α | Last-5 Avg Reward | Max Reward |
|---|---|---|---|---|---|
| run1 (baseline) | 1e-5 | 0.9 | 16/32 | 0.400 | 2.0 (once) |
| run2 (higher LR) | 5e-5 | 0.9 | 16/32 | 1.300 | 2.0 |
| run3 (best) | 5e-5 | 0.7 | 32/64 | **1.500** | 2.0 |

The best model was run3. The higher learning rate was clearly the most impactful change — run1 with lr=1e-5 barely improved over 100 steps, while run2 and run3 both climbed to 2.0 early. Lowering temperature from 0.9 to 0.7 in run3 further stabilized convergence by focusing sampling on higher-probability outputs. Run3 reached the maximum reward of 2.0 by step 21 and hit it multiple times.

## Evaluation

The best GRPO model (run3) was evaluated against the base zero-shot model on 156 held-out BUSI images:

| Model | Accuracy | Format | Benign | Malignant | Normal |
|---|---|---|---|---|---|
| Base (zero-shot) | 21.8% (34/156) | 31.4% | 36.9% | 5.7% | 2.7% |
| GRPO-trained | **48.1% (75/156)** | **100.0%** | **53.6%** | **85.7%** | **0.0%** |

GRPO more than doubled overall accuracy. The improvement was sharpest for malignant cases (5.7% → 85.7%) and solid for benign (36.9% → 53.6%). The format reward worked exactly as intended: the base model only used the "Answer:" format 31.4% of the time, while the GRPO model used it 100%.

The failure case was striking: the GRPO model got **0/37 correct on normal images**, classifying every one as benign or malignant. The base model was also poor on normal (2.7%), so this was a pre-existing weakness — but GRPO may have reinforced it by never generating a correct "normal" prediction during training, leaving the model with no positive signal to learn from for that class.

## Reflection

The comparison with HW3 SFT was instructive. SFT converged faster and more smoothly — every token in the target provides a direct gradient. GRPO was noisier, and the accuracy reward component stayed near 0 during training for all runs, meaning the model was learning primarily from format compliance rather than from getting classification answers right. The test-time improvement came through a shift in output structure and pattern, not from a dense accuracy signal during training.

The bigger takeaway is about when RL is the right tool. GRPO is most effective when the model already has latent ability and needs better calibration. For BUSI ultrasound, where the base model starts with limited domain knowledge, there isn't much correct behavior for GRPO to amplify — which is why SFT's direct supervision worked better overall. The normal-class collapse is a concrete example: GRPO couldn't fix a capability gap, it could only calibrate what was already there.

## Results and Visualizations

### Run 1 Baseline Training Summary
![Run 1 training summary](imgs/Screenshot%202026-05-07%20at%2012.18.06%20PM.png)
*Run1 baseline (lr=1e-5, LoRA r=16): reward barely moved — last-5 average of 0.400, same as the first 5 steps. The slow learning rate meant the model rarely found a differential training signal within 100 steps.*

### Hyperparameter Comparison
![Hyperparameter sweep results](imgs/Screenshot%202026-05-07%20at%2012.18.17%20PM.png)
*Run2 vs. run3 comparison. Both used lr=5e-5; run3 additionally lowered temperature to 0.7 and increased LoRA r to 32. Run3's last-5 avg reward of 1.500 beat run2's 1.300, and it finished 3.7 minutes faster (11.3 vs 15.0 min).*

### Final Evaluation
![Evaluation results](imgs/Screenshot%202026-05-07%20at%2012.18.26%20PM.png)
*GRPO-trained vs. base model on 156 held-out BUSI images. GRPO reached 48.1% (75/156) with 100% format compliance. The improvement was sharpest for malignant (85.7%) while normal collapsed to 0/37.*
