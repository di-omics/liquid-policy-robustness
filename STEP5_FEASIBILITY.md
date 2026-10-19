# Step 5 feasibility: LFM2.5-VL-450M as a camera-to-torque policy

Checked 2026-10-19 on the M5 MacBook Air (24 GB, 10-core GPU, fanless), macOS 27.0.1,
torch 2.14.1 (MPS), transformers 5.20.1, MLX 0.32.4, mlx-vlm 0.7.8, gymnasium 1.4.0,
mujoco 3.16.0. All benchmarks ran in a scratch environment outside this repo, on frames
rendered from training seeds 0-3 with the camera from `collect.py`. Nothing here runs the
noise sweep, touches the TODO in `common.py`, or trains a policy to completion: the
training numbers below are speed and memory only, measured with random targets.

## Short answer

- **Loading and inference fit easily.** The model loads in about 2 s and takes about 1 GB of
  GPU memory in bf16.
- **One forward pass per frame takes 22-40 ms** (MLX compiled vs PyTorch), so the laptop
  cannot run the policy at Reacher's 50 Hz in real time. That does not matter in
  simulation, where the environment waits for the policy. Batched across the 100
  evaluation seeds, a full sweep (6 noise levels) takes about 13 minutes per trained model.
- **Designs that keep the model frozen are cheap on the laptop.** One pass of about 13
  minutes stores the model's features for all 30,000 demonstration frames (25,000 training,
  5,000 validation), and small heads then train at about 22,000 frames/s.
- **Fine-tuning is the bottleneck.** The smallest design that adapts Liquid AI's own part of
  the model (LoRA on the LFM2 backbone, vision tower frozen) trains at 14 frames/s in
  PyTorch and 22 frames/s in MLX, sustained over 5 minutes: 19-30 minutes per pass over the
  25,000 training frames. With an assumed 20 passes per run and the repo's 5 runs, that is
  about 37 h (MLX) to 53 h (PyTorch) of laptop GPU time, and about 1.6x that with two
  frames per input.
- **Recommendation.** Do the frozen-feature stage and one short LoRA pilot on the laptop.
  For the five LoRA runs, rent one 24 GB GPU (RTX 4090 or L40S) for an afternoon: an
  estimated $5-20 in all, setup included. Nothing is rented yet. The free alternative is
  five overnight MLX runs on the laptop, which needs an MLX version of the training code.

## 1. Loading (measured)

Each row is a fresh Python process; "load" is after the imports, which take another
8-10 s on their own.

| build | load | GPU memory after load |
|---|---|---|
| transformers, bf16, MPS (`LiquidAI/LFM2.5-VL-450M`) | 1.6-2.1 s | 0.90 GB allocated, 1.29 GB held by the MPS driver |
| transformers, fp32, MPS | 2.0 s | 1.80 GB allocated, 2.53 GB held |
| MLX bf16 (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`) | 0.45-0.53 s | 0.90 GB |
| MLX 8-bit (`LiquidAI/LFM2.5-VL-450M-MLX-8bit`) | 0.51 s | 0.56 GB |

Parameter count, from the loaded transformers model: 448.7 M in total, of which the
vision tower is 85.8 M, the projector 8.4 M and the language model 354.5 M (67.1 M of it
the token embedding, shared with the output layer). The vision tower is SigLIP2 NaFlex
(base, 16-pixel patches), originally a Google model; the language backbone is Liquid AI's
LFM2.5-350M, the part with the 10 convolution and 6 attention blocks. That split matters
for the design in section 3.

Each download (transformers bf16, MLX bf16) is 0.9 GB.

## 2. Frames and one forward pass (measured)

**Frames.** `gym.make("Reacher-v5", render_mode="rgb_array", default_camera_config=TOP_DOWN,
width=256, height=256)` with `TOP_DOWN` exactly as in `collect.py`. Rendering takes 0.78 ms
per frame at 256x256 and 1.27 ms at the default 480x480; one physics step takes 0.06 ms.
At 256x256 the frame shows about 0.62 m of arena, 2.4 mm per pixel, and the target is a
dot about 7 pixels across. Arm, fingertip and target are all clearly visible.

**What the model sees.** The processor turns a 256x256 frame into 256 vision patches and
64 image tokens (78 tokens with the chat template and a short prompt). Smaller frames are
upscaled to the same 64 tokens, the configured minimum. A 480x480 frame becomes 900 patches
and 225 tokens (239 in all). The stock processor pads every image to 1024 patches, so at
256x256 the vision tower processes 4x as many patches as the image has. Cutting the
padding (and doing the patchify on the GPU, which matches the processor's pixel values
exactly) gives the same output as the stock path to a relative error of 2e-6 in fp32 and
makes a batch of 32 frames 2.6x faster. All numbers below use the unpadded path.

**Time per forward pass** (vision tower, projector and the LFM2 backbone up to its last
hidden state; the torque head itself is negligible):

| backend, precision, frame size | batch 1 | batch 8 | batch 32 | batch 100 |
|---|---|---|---|---|
| PyTorch MPS, bf16, 256 px | 39.7 ms (25 frames/s) | 41 frames/s | 41 frames/s | 39 frames/s |
| PyTorch MPS, fp32, 256 px | 112 ms (9 frames/s) | 12 frames/s | 10 frames/s | |
| PyTorch MPS, bf16, 480 px | 108 ms (9 frames/s) | 9 frames/s | | |
| MLX bf16, 256 px, `mx.compile` | 21.9 ms (46 frames/s) | 34 frames/s | 36 frames/s | 37 frames/s |
| MLX 8-bit, 256 px, `mx.compile` | 46.0 ms (22 frames/s) | 38 frames/s | 33 frames/s | 39 frames/s |

The vision tower alone takes 10.5 ms (MLX) to 12.8 ms (PyTorch) for one frame, and
PyTorch runs it at about 93 frames/s in batches. Outputs agree across backends: against the
PyTorch fp32 result, PyTorch bf16 differs by 1.3%, MLX bf16 by 1.6% and MLX 8-bit by 2.7%
(relative error of the last hidden state over 8 frames).

Batching barely helps: about 40 frames/s is the ceiling on both backends. One frame costs
about 92 GFLOP (about 46 in the vision tower over 256 patches, about 45 in the LFM2
backbone over 78 tokens), so 40 frames/s is about 3.7 TFLOPS. For comparison, a plain bf16
matrix multiply on this GPU ran at 11.8 TFLOPS at 2048x2048 and 4.9 TFLOPS at 4096x4096,
about the same on both backends, so better kernels might buy 2-3x but not 10x. fp32 is
3.5-4x slower, 480 px about 4x slower, and 8-bit weights save memory but not time. **Use
bf16 at 256x256.**

**Closed-loop rollouts.** Reacher steps every 0.02 s. The best single-frame latency here,
about 23 ms including rendering and preprocessing (MLX compiled), runs the policy at about
43 Hz, just short of real time. In simulation that is irrelevant: the environment pauses
while the policy thinks, so the only question is total time. Run the 100 evaluation seeds
as 100 environments side by side and send one batch of 100 frames per step: 50 steps x
(2.6 s forward + 0.08 s rendering) is about 2.2 minutes per noise level, about 13
minutes for all 6 levels, and a little over an hour for 5 trained models. Closed-loop
evaluation on the laptop is practical. Noise added to the images costs nothing extra; it
is applied before the forward pass. Real-time 50 Hz would matter only on a physical arm,
and there the frozen vision tower with a small head (about 11-13 ms per frame) would fit.

## 3. A design for continuous torques

Three facts shape the design.

1. **Liquid AI's part of the model is the LFM2 backbone.** The vision tower is SigLIP2. A
   policy built from the vision tower plus a head would test Google's encoder, not Liquid
   AI's model. To answer the step-5 question, the image has to pass through the LFM2
   backbone.
2. **One frame shows positions, not speeds.** The step-2 students read the joint speeds
   directly (`obs[6:8]`). A camera policy that needs speed has to get it
   from two frames or from memory across steps. Whether the teacher's torques depend on
   speed is left to Q1 in `NOTEBOOK.md`.
3. **The output is two numbers in (-1, 1), not text.** Writing torques as text tokens
   ("0.42, -0.17") would take several decoding steps per frame and round the values. A
   regression head on the backbone's last hidden state gives both torques in one forward
   pass.

**Stage A: frozen model, small heads (laptop, under an hour).** Run the frozen model once
over all 30,000 frames (about 13 minutes at 40 frames/s) and store two features per frame:
the hidden state at the last position and the mean over the image-token positions, 1,024
numbers each (245 MB for both in fp32). Train small heads on them: an MLP on one frame, and
LSTM and CfC heads over each episode's sequence of features. A head on cached features
trains at about 22,000 frames/s (measured for an 82,000-parameter head on cached image
tokens), so 300 passes take minutes. This shows whether the frozen model already carries
enough information to control the arm, and it runs the repo's memory-versus-liquid
comparison on image features at almost no cost. The heads' first layer grows with the
1,024-number input, so keeping them near 5,000 parameters needs a shared projection to a
few numbers first.

**Stage B: LoRA on the LFM2 backbone (the smallest design that adapts Liquid AI's part).**

- Input: the current frame, and the previous frame if speeds matter, at 256x256, with a
  fixed short prompt.
- Vision tower and projector frozen. Their 64 image tokens per frame are computed once and
  cached (3.9 GB in bf16 for 30,000 frames).
- LoRA rank 16 on every attention, convolution-projection and feed-forward layer of the
  backbone: 6.0 M trainable parameters, 1.7% of the backbone.
- Head: a linear layer from 1,024 numbers to 2, then tanh, on the hidden state at the last
  position, so it can reach the full torque range like the step-2 students.
- Loss: mean squared error against the teacher's torques, as in `train.py`. Keep the
  checkpoint with the lowest error on validation seeds 500-599.

Not recommended for a first pass: fine-tuning all 449 M parameters (fp32 weights plus
AdamW state come to about 5.4 GB, and 25,000 frames are few for that many parameters);
end-to-end LoRA through the vision tower (2.5x slower, see below; worth trying only if the
frozen encoder misses detail); torques as text.

**Training speed on the laptop (measured).** Batch 16 unless noted, 256x256, bf16 base
weights, random targets. "Pass" is one epoch over the 25,000 training frames.

| design | trainable parameters | frames/s | GPU memory | one pass |
|---|---|---|---|---|
| head on cached image tokens (batch 256) | 0.08 M | 22,000 | 2.4 GB | 1 s |
| full fine-tune of vision tower + projector + head (fp32 weights, bf16 autocast) | 94.3 M | 12.8 | 4.9 GB | 33 min |
| **Stage B, PyTorch** (LoRA on LFM2, cached image tokens) | 6.0 M | 15.0 over 1 min, 13.9 over 5 min | 3.6 GB | 30 min |
| **Stage B, MLX** (same design, mlx-vlm LoRA layers) | 6.0 M | 22.4 over 1 min, 22.2 over 5 min (20.1 in the last minute) | 3.3 GB | 19 min |
| Stage B, vision tower re-run every step without gradients (needed to train with image noise) | 6.0 M | 12.7 | 3.7 GB | 33 min |
| LoRA on vision tower and LFM2, end to end | 8.7 M | 5.5 | 7.9 GB | 75 min |
| Stage B with two frames per sample (128 image tokens), PyTorch | 6.0 M | 8.6 | 6.0 GB | 48 min |

Throughput stops growing past batch 8, as for inference. Over 5 minutes the PyTorch run
did not slow down, and the MLX run lost about 10% in its last minute.

## 4. Laptop or rented GPU

**How much training.** The step-2 students get 300 passes; a pretrained model adapted with
LoRA usually needs tens. This estimate assumes 20 passes per run (500,000 frames), to be
replaced by what the validation curve shows. The repo's protocol is 5 training runs per
design. Checking all 5,000 validation frames after every pass adds about 2 minutes per pass
at the inference rate.

| where | Stage B, one frame | Stage B, two frames |
|---|---|---|
| laptop, MLX (about 21 frames/s) | about 7.3 h per run, 37 h for 5 | not measured; likely about 1.6x the one-frame time |
| laptop, PyTorch (13.9 frames/s) | about 10.7 h per run, 53 h for 5 | about 17 h per run, 86 h for 5 |
| rented RTX 4090, 24 GB (estimated 300-500 frames/s) | 1.5-2.5 h for all 5 runs | about 2.5-4 h for all 5 |
| rented A100, 80 GB (estimated 600-900 frames/s) | 0.8-1.3 h for all 5 runs | about 1.3-2 h for all 5 |

The rented-GPU speeds are estimates, not measurements. One Stage B training frame costs
about 135 GFLOP (3x the 45 GFLOP forward pass through the backbone). An RTX 4090 peaks at
165 TFLOPS in bf16 and an A100 at 312; a model this small typically uses 25-40% of that.
Counted the same way, the laptop reaches about 1.9 TFLOPS on this design in PyTorch and
3 TFLOPS in MLX, so a 4090 should be roughly 20-35x faster than the laptop in PyTorch.

**Cost.** On-demand prices in mid-October 2026 were about $0.34-0.89 per hour for an RTX
4090 (RunPod community and secure cloud), $0.79-1.09 for an L40S, $1.39-1.59 for an A100
80 GB and $2.69-3.49 for an H100. The five one-frame runs need about $1-2 of 4090 time.
Setting up the environment (MuJoCo renders headless with `MUJOCO_GL=egl`; rendering all
30,000 frames takes under a minute), a pilot run, mistakes and the closed-loop noise sweep
bring a realistic budget to $5-20 for step 5, and under $30 even if the speed estimates
are 3x optimistic. An H100 is not needed: the model is too small to use it well.

**Decision.** The laptop is enough for everything except the five fine-tuning runs, where
it is possible but slow: about two days of continuous GPU time in PyTorch, or five
overnight runs in MLX, on a fanless laptop. Suggested
order:

1. Stage A on the laptop (free, under an hour of compute).
2. One short Stage B pilot on the laptop, about 2 passes (1-2 h), to check that the loss
   falls and the pipeline works end to end.
3. The five Stage B runs on one rented RTX 4090 or L40S.
4. The closed-loop noise sweep on the laptop (about 13 minutes per trained model) or on
   the same rented machine.

## Notes and limits

- No design has been trained to convergence, so nothing here says whether a 450M VLM can
  imitate the teacher as precisely as the 5,000-parameter students. The training numbers
  are speed only. A short real training run of the recommended design is the next check.
- Training speeds were measured on 200 frames held in GPU memory, cycled with random
  targets. Loading 25,000 real frames from disk adds little: the cached image tokens for
  30,000 frames are 3.9 GB in bf16, and raw 256x256 frames are 5.9 GB uncompressed.
- The laptop has no fan. The 5-minute sustained runs in section 3 are the only evidence on
  throttling; a 7-11 hour run could slow further.
- GPU rental speeds are estimates from FLOP counts and typical utilization, not
  measurements. Prices are on-demand list prices from mid-October 2026 and change often.
- A cheaper input is available. The model card suggests `min_image_tokens=32`; with it, a
  192x192 frame (3.2 mm per pixel) becomes 36 image tokens and 50 tokens in all. Inference
  then runs at 57 frames/s at batch 32 instead of 41, and 25.6 ms instead of 39.7 ms for one
  frame (PyTorch, bf16): 1.4x faster, not 2x, because the prompt and fixed costs stay.
  Training should speed up by a similar factor (not measured). Whether 192 px is sharp
  enough for 2 cm accuracy is a question for a training run.
- The benchmark scripts are throwaway timing code and are not in this repo.
