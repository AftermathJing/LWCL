# Architecture contract

The thesis is the design authority. The reconstruction uses one canonical tensor contract throughout:

`features: [batch, time, receivers, features_per_receiver]`

The paper-aligned path is:

1. Intel 5300 CSI parsing and RSSI extraction.
2. Reference-antenna selection, conjugate differential CSI, 2-60 Hz filtering.
3. Complex PCA, STFT Doppler representation, RSSI/Doppler/differential-CSI fusion.
4. Per-receiver standardization and time alignment, yielding 49 features by default.
5. Channel Attention using temporal GAP/GMP and receiver-feature fusion.
6. Hierarchical spatio-temporal encoder:
   - projection plus absolute sinusoidal position encoding;
   - overlapping local RoPE windows;
   - learned convolutional window aggregation;
   - global RoPE Transformer.
7. Adapter projection plus shallow RoPE Transformer.
8. A continuous-embedding classifier backbone:
   - `tiny` for mandatory smoke tests;
   - Qwen2.5 sequence classification with LoRA for the paper route.

The tiny backend is a test instrument, not a replacement research claim. A Qwen run is not permitted until the tiny route proves forward/backward, evaluation, save, resume, and test evaluation.
