# LWCL thesis implementation specification

This document condenses the implementation-relevant content of the undergraduate thesis **Research and Implementation of Wireless Signal Correlation Learning Framework Based on Large Language Model**. It is the fast reference for reconstruction work; the thesis remains the authority when a detail is not covered here.

## 1. Research objective

LWCL is intended to map heterogeneous wireless observations to task semantics while modelling correlations inside the signal rather than relying only on handcrafted physical features.

The target capabilities are:

- accuracy comparable to or better than the wireless-sensing state of the art;
- parameter-efficient training with most LLM parameters frozen;
- improved transfer across environments, positions and orientations;
- a reusable path from physical wireless measurements to semantic task outputs.

The thesis validates one concrete task: six-class Wi-Fi CSI gesture recognition on Widar3.0. Claims about other signals and tasks are future directions, not completed experiments.

## 2. Signal-correlation definition

LWCL models three types of correlation:

1. **Dimension correlation**: coupling among frequency, amplitude, phase, RSSI, CSI and Doppler features.
2. **Spatial correlation**: related observations from different antennas, receivers and locations.
3. **Temporal correlation**: ordered and long-range structure across the stages of an action.

The conceptual mapping is:

```text
multi-receiver physical signal tensor
  -> correlation-aware feature fusion
  -> environment-reduced joint embedding
  -> task-specific semantic output
```

## 3. Three-stage architecture

### 3.1 Data preprocessing

Input sources used by the thesis implementation:

- Intel 5300 CSI complex measurements;
- per-antenna RSSI and total RSSI;
- Doppler spectrum derived from differential CSI.

Paper-aligned processing sequence:

1. Decode Intel 5300 `.dat` bfee records and 30 subcarriers.
2. Extract complex CSI and RSSI for every receiver.
3. Select a reference antenna pair using a mean-to-variance criterion.
4. Remove static amplitude baselines while retaining phase.
5. Use conjugate multiplication against the reference to reduce hardware phase offset and static reflection effects.
6. Apply a 60 Hz low-pass and 2 Hz high-pass Butterworth filter.
7. Use Complex PCA on differential CSI.
8. Use STFT or CWT to obtain motion-related Doppler features.
9. Aggregate and standardize RSSI, Doppler and differential-CSI features per receiver.
10. Align receiver timelines, pad/truncate sequences and construct masks and position IDs.

Default reconstructed feature vector per receiver and time step:

| Feature family | Dimension | Notes |
|---|---:|---|
| RSSI | 4 | three antenna RSSI values plus total RSSI |
| Doppler spectrum | 25 | aggregated from approximately -60 Hz to 60 Hz |
| Differential CSI | 20 | ten complex PCA components split into real and imaginary parts |
| Total | 49 | canonical `F` dimension |

Canonical implementation tensor shape:

```text
[B, T, N, F]
B: batch
T: time
N: receivers/sensors, paper experiment default 6
F: per-receiver features, paper reconstruction default 49
```

The old source alternated between `[B,T,N,F]` and `[B,T,F,N]`. The reconstruction fixes `[B,T,N,F]` as the only public contract.

### 3.2 Signal embedding

#### Channel Attention

Purpose:

- dynamically weight receiver-feature combinations;
- combine stable temporal statistics and transient events;
- fuse multiple receivers while reducing redundant dimensions.

Design:

1. Temporal global-average pooling extracts stable patterns.
2. Temporal global-max pooling extracts transient peaks.
3. A convolutional MLP produces a receiver-feature weight matrix.
4. The original input is reweighted.
5. A convolution covering the entire feature dimension fuses all receiver channels.
6. Layer normalization produces `[B,T,M]`.

Paper/default value: `M=64`.

#### Hierarchical Spatio-Temporal Encoder (HSTE)

Purpose:

- suppress short-term asynchronous/multipath interference;
- model local action fragments and global action progression;
- shorten the sequence before the LLM;
- reduce the dimensional gap between wireless and LLM embeddings.

Design:

1. Project `M=64` features to `C=128`.
2. Add absolute sinusoidal position encoding.
3. Split into overlapping windows, default size 8 and stride 4.
4. Apply a local RoPE Transformer inside every window.
5. Aggregate each window with a learned 1D convolution.
6. Increase representation dimension to `C'=256`.
7. Apply a global RoPE Transformer across window embeddings.

Paper/default layers:

- local: one layer, four heads;
- global: two layers, eight heads;
- dropout: 0.1.

### 3.3 Correlation learning

#### Adapter

The Adapter maps the HSTE output dimension to the selected LLM hidden size and applies a shallow RoPE Transformer for distribution alignment.

Paper/default design:

- input: `C'=256`;
- nonlinear projection to LLM hidden size;
- two Transformer layers;
- eight attention heads;
- GELU, LayerNorm and dropout.

#### LLM and PEFT

The thesis implementation reports Qwen2.5-7B-Instruct as the base model.
The remote reconstruction expects Qwen2.5 weights under user-home directories such as `/home/wj/Qwen2.5-0.5B-Instruct`, `/home/wj/Qwen2.5-1.5B-Instruct`, `/home/wj/Qwen2.5-3B-Instruct` and `/home/wj/Qwen2.5-7B-Instruct`. The smaller models are exploration backends; Qwen2.5-7B-Instruct is the thesis-aligned route.

Wireless embeddings are passed through `inputs_embeds`; they are not converted into text tokens. The intended tuning policy is:

- freeze the main LLM backbone;
- inject LoRA into attention projections;
- keep feed-forward layers frozen;
- train normalization layers to adapt to wireless-feature distributions;
- train the task classification head;
- use a task-specific output head for each downstream task.

Reported thesis parameter scale:

- approximately 6.75 billion parameters in the full framework;
- approximately 220 million trainable parameters overall (3.31%);
- approximately 5.23 million fine-tuned Qwen parameters reported separately.

These numbers must be re-measured from the resolved model and configuration; they are not hard-coded acceptance criteria.

## 4. Dataset and sampling protocol

Widar3.0 collection described in the thesis:

- one transmitter and at least three receiver nodes in the general setup;
- the selected experiment uses six receiver observations per action group;
- Intel 5300 NICs at 5.825 GHz/channel 165;
- 1000 packets per second;
- six gestures: push/pull, sweep, clap/strike, slide, circle and zigzag;
- 14 selected volunteers from the larger dataset description;
- five positions and five orientations;
- five repetitions;
- approximately 63,000 receiver-level samples reported for the selected scope.

The main reported evaluation is narrower:

- data collected on 2018-11-03;
- nine volunteers, IDs 5 and 10-17;
- 648 action groups;
- six receiver files per group, 3,888 receiver-level files;
- three random subsets of 216 groups each.

Reconstruction policy:

- keep `sample_id` at the action-group level so receiver files cannot cross splits;
- primary evaluation uses subject-disjoint splits;
- strict cross-environment evaluation uses environment-disjoint splits;
- random sample splits may be generated only as an explicitly labelled thesis-reproduction protocol;
- class-balanced sampling is allowed only in the training split;
- validation and test loaders never use weighted resampling.

## 5. Evaluation contract

Required metrics:

- classification accuracy;
- macro precision;
- macro recall;
- macro F1;
- per-class precision, recall and F1;
- confusion matrix;
- subgroup accuracy by subject, environment, position and orientation.

Reported thesis results:

| Model | Accuracy |
|---|---:|
| LWCL | 93.17% +/- 1.93% |
| Widar3.0 | 89.5% +/- 1.4% |
| CNN + LSTM | 40.6% +/- 4.5% |
| CNN + GRU | 39.6% +/- 8.0% |
| LSTM | 21.8% +/- 6.7% |
| CNN | 18.3% +/- 7.1% |

Reported transfer figures:

- Room1/training environment: 93.17%;
- Room2/unseen hall: 80.43%;
- Room3/unseen office: 76.65%;
- position few-shot average: 89.37%;
- orientation few-shot average: 85.31%.

Important interpretation:

- the thesis value 83.42% averages Room1, Room2 and Room3; the average of the two unseen rooms alone is 78.54%;
- position and orientation results use target-domain few-shot tuning and are not zero-shot results;
- the thesis mixes the Chinese terms for accuracy and precision in the metric section, so the reconstruction records both explicitly;
- random sample splitting does not establish unseen-subject generalization.

## 6. Required ablations before restoring the research claim

The original thesis does not provide enough evidence to isolate the LLM contribution. A restored project should compare at least:

1. preprocessing + simple classifier;
2. Channel Attention + HSTE without an LLM;
3. full encoder + small randomly initialized Transformer;
4. full encoder + frozen Qwen without LoRA;
5. full LWCL with Qwen LoRA;
6. removal of Channel Attention;
7. removal of local-window encoding;
8. removal of the Adapter Transformer.

Paper-scale performance is not considered reproduced until the baselines use comparable preprocessing, splits, optimization budgets and early-stopping rules.

## 7. Known thesis ambiguities resolved by this repository

| Ambiguity | Reconstruction decision |
|---|---|
| Tensor dimension order differs between text and old code | Public input is always `[B,T,N,F]` |
| Accuracy and precision terminology is mixed | Report accuracy and precision separately |
| Full 63,000 sample description differs from 3,888-file main evaluation | Record both scopes in the run metadata |
| Environment average includes the training room | Report seen and unseen-room averages separately |
| Exact Qwen hidden dimension was inconsistent in old config | Read hidden size from the Hugging Face model config |
| Word embedding replacement was treated as mandatory | Use the supported `inputs_embeds` path; do not destructively replace embeddings |
| Checkpoint behavior was unspecified | Save optimizer, scheduler, scaler, RNG, config and step; Qwen checkpoints store trainable weights only |

## 8. Code mapping

| Thesis component | Reconstructed implementation |
|---|---|
| Intel 5300 decoding | `src/lwcl/data/intel5300.py` |
| CSI/RSSI/Doppler preprocessing | `src/lwcl/data/preprocessing.py` |
| Dataset, padding and balanced sampling | `src/lwcl/data/dataset.py` |
| Leakage-resistant splits | `src/lwcl/data/splits.py` |
| Channel Attention | `src/lwcl/models/channel_attention.py` |
| RoPE attention | `src/lwcl/models/rotary.py` |
| HSTE | `src/lwcl/models/hste.py` |
| Adapter | `src/lwcl/models/adapter.py` |
| Tiny/Qwen backends | `src/lwcl/models/backbones.py` |
| End-to-end LWCL | `src/lwcl/models/lwcl.py` |
| Training and emergency checkpointing | `src/lwcl/training/trainer.py` |
| Metrics and subgroup evaluation | `src/lwcl/training/metrics.py` |
| Remote environment and smoke | `scripts/bootstrap_remote.sh`, `scripts/remote_smoke.sh` |

## 9. Current evidence boundary

The repository reconstructs the paper-described workflow, but it does not yet claim that the reported 93.17% result has been reproduced. That claim requires:

1. real Widar3.0 data and a versioned manifest;
2. successful raw preprocessing inspection;
3. remote forward/backward, evaluation, checkpoint, resume and emergency-checkpoint smoke tests;
4. controlled baseline and ablation runs;
5. subject-disjoint and environment-disjoint evaluation;
6. archived resolved configs, logs, metrics and checkpoint metadata.
