# Signal-v2 architecture contract

## Scope

LWCL-v2 is a wireless-signal representation model. LLM, Adapter, LoRA, prompts, text tokens and language-space claims are outside the project boundary.

## Data flow

```text
raw six-receiver Intel 5300 files
  -> per-receiver quality metrics
  -> retain 6/6 or 5/6 valid receivers
  -> timestamp-aware P1 STFT at 20 Hz
  -> separate RSSI [T,6,4], Doppler [T,6,25], differential CSI [T,6,10,2]
  -> feature-family stems (SiLU)
  -> shared receiver MLP (SiLU)
  -> mean/max/std/quality-attention receiver fusion
  -> first difference + depthwise temporal convolutions (SiLU)
  -> local window RoPE Transformer (GELU FFN)
  -> attentive+mean window token
  -> global physical-time RoPE Transformer (GELU FFN)
  -> attentive mean/std/max statistics
  -> GELU classification head
```

## Position contract

The default does not add an absolute sinusoidal vector to signal content.

- Local RoPE positions reset to `0..W-1` in every window.
- Every processed NPZ stores `frame_times_ms`.
- Global window positions are the original short-window center times, divided by `rope_time_unit_ms=50` before RoPE.
- The actual millisecond values are retained in model outputs for auditing.
- A zero-initialized normalized-phase residual exists only in the `position_c_phase.yaml` ablation.

This preserves ordering while avoiding direct binding between signal content and an absolute frame embedding.

## Activation contract

- SiLU: RSSI, Doppler, differential-CSI stems; feature fusion; shared receiver encoder; temporal convolution.
- GELU with tanh approximation: local/global Transformer FFNs, window pooling projection and classifier hidden layer.
- Softmax: competitive receiver and temporal attention.
- No activation after final classification logits.
- Parameter-matched SwiGLU is available only through `activation_c_swiglu.yaml`.

## Quality contract

A receiver is valid when:

```text
packet_count >= stft_window + 4 * stft_hop
packet_count / group_median >= 0.60
timestamp_gap_ratio <= 0.10
```

Six or five valid receivers are retained. Four or fewer cause sample rejection. Invalid receiver tensors are zero-filled and excluded by `receiver_mask`; they never shorten valid receivers.

## Selection contract

The primary checkpoint score is:

```text
0.7 * validation_macro_f1 + 0.3 * worst_validation_subject_macro_f1
```

The test split is evaluated only after selecting a checkpoint on validation subjects.
