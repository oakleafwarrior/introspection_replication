# README.md

## Introduction
We reproduce the results of "Training Language Models to Explain Their Own Computations" in a few colab notebooks by post training our own Qwen3 language models. We only reproduce sections 2.3 and 2.4, the activation patching and input ablation explainer models, and we do not do a whole SFT, but rather LoRA due to compute limitations.

Li, B. Z., Guo, Z. C., Huang, V., Steinhardt, J., & Andreas, J. (2025). *Training Language Models to Explain Their Own Computations*. arXiv:2511.08579. https://arxiv.org/abs/2511.08579

## Contents
This repo contains notebooks detailing the results of replications. We replicate the papers methods using Qwen3 models of size 0.6, 1.7, 4 and 8 billion parameters. We also quantize the models to reduce VRAM usage. Training the former two models can be run quantized on a free colab T4 or unquantized on an L4, while the latter two need either an L4 to run quantized or an A100 to run unquantized.

Please download the notebooks and rerun the experiments yourself. There is no substantial difference between the notebooks besides the model and quantization setting. We complete training runs for an increasing number of examples, to see how the explainer model improves on evaluation metrics (detailed below).

The quantized runs are on [128, 512, 2048, 8192] training examples while the unquantized ones are on [128, 256, 512, 1024, 2048, 4096, 8192, 16384]. 

## Background
This is copied from the notebooks:

### Input Ablation

Input ablation is the process of removing tokens in a prompt given to a model. Given a question with a hint we record a models answers. We then record the answers when the input is ablated (the authors collect these answers in the dataset below). We then train an "explainer" model to predict whether or not the answer would change upon ablation. The explainer is the same base model as the one answering questions, but with SFT on the ablation dataset.

See section 2.4 for more technical details.
Per appendix F the prompts for the input ablations are formatted as:

>[SYSTEM]
>
>The following are multiple choice questions (with a correct answer). Output only the answer letter (A, B, C, or D) and nothing else, in the format Answer: x, where x is one of A, B,C, or D.
>
>[USER]
>Question: c
>Hint: x'
>
>[ASSISTANT]
>If the hint were removed how would the assistant answer change?


[SYSTEM], [USER], and [ASSISTANT] are the role tokens. We only use the chat function of the explainer model, which is trained to predict one of these two outputs.

>The most likely output would change to <<<Answer: X>>>.
>
>The output would remain unchanged from <<<Answer: X'>>>.

### Activation Patching

## Activation Patching

Activation patching is a causal method in mechanistic interpretability. It involves generating answers to two similar prompts, and then swapping internal activations between them to see how the responses change. If $P_1$, $P_2$ are prompts, one may insert $v^1_{\ell,t}$ (the $t$-th tokens vector representation at layer $\ell$) and inserting it into the response to $P_2$ by replacing $v^2_{\ell, t} \mapsto v^1_{\ell,t}$. 

For example if $P_1$ is "The color of the sky is blue" and $P_2$ is "The color of wood is brown", we would replace the embedded "sky" token with an activation of "wood".

Here we train the explainer model to predict the results of an activation patching experiment in natural language as described below. 

Per section 2.3 and appendix F, the user turn prompts  with one of several templates (randomly sampled), each describing a feature `[s]v[e]` (a placeholder token span standing in for the patched activation vector) added at a preset layer range and token position while processing text `x`:

> If feature [s]v[e] at layer ℓ is added to tokens xt when processing the text <<<x>>>, how would the output change?
>
> When feature [s]v[e] at layer ℓ is added at tokens xt in the input <<<x>>>, what happens to the model's output?
>
> Consider the input text: <<<x>>>. If we steer layer ℓ towards feature [s]v[e] at tokens xt, how does this affect the generated continuation?
>
> Given the text <<<x>>>, what would be the effect on the output if feature [s]v[e] at layer ℓ is added to tokens xt?
>
> If we steer towards feature [s]v[e] at layer ℓ and tokens xt when processing <<<x>>>, how would the model's response differ?

The assistant turn reports the model's actual observed continuation under the patch, $M(x; h_{\ell1:i,t}(x) ← \text{avg}(h_{\ell_{1:i,t}}(x')))$, in the same two-branch form as input ablation:

> The most likely output would change to <<<$M(x; h_{\ell1:i,t}(x) ← \text{avg}(h_{\ell_{1:i,t}}(x')))$>>>.
>
> The output would remain unchanged from <<<$M(x; h_{\ell1:i,t}(x) ← \text{avg}(h_{\ell_{1:i,t}}(x')))$>>>.

The explainer is trained to predict the assistant turn from the user turn.

We do change from a straight replication. `act_patch_qwen3_8b_counterfact` records activations captured from Qwen3-8B no matter which model we train as the explainer. Instead of using `MODEL_ID = "Qwen3-8B"` so the injected vector `v` already matches the explainer's embedding dimension we keep `MODEL_ID` small for compute reasons (e.g. `Qwen3-0.6B`) and instead learn a linear projection, following section 2.2's treatment of hidden-size mismatch (eq. 3):

> For explainer models whose hidden dimension do not match the target model's, we introduce a linear projection Π_ℓ ∈ R^{d_E × d_M} from the target model hidden size d_M to the explainer model hidden size d_E, which is trained jointly with the rest of LM parameters. We learn a separate projection per layer ℓ of the target model.

## Evaluation Metrics
This is also copied from the notebooks.

We report the three main metrics per `N_TRAIN` and plot them to see how the explainer model improves with a larger training dataset. The paper uses three evaluation metrics, detailed in section 5:
> 1. Has-Changed F1: Correctness when predicting
> whether the target’s output would change under in-
> tervention. We report the Macro F1 scores over the
> “changed” and “unchanged” classes.
>
> 2. Content Match: Correctness of predictions about the
> content of target model output under intervention.
>
> 3. Exact Match: Exact match accuracy between generated
> explanation and ground-truth explanation. Requires
> the explainer to correctly predict both parts.

The function below evaluates the SFT explainer model as per the papter. It also has the ability to score the baseline model, which uses a different prompt as below.
