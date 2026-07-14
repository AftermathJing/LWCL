# Remote runbook

Target server: `wj@10.69.216.119`. Training and deployment are remote-only.

After the GitHub repository is available:

```bash
ssh wj@10.69.216.119
git clone https://github.com/AftermathJing/LWCL.git
cd LWCL
bash scripts/bootstrap_remote.sh
bash scripts/remote_preflight.sh
bash scripts/remote_smoke.sh
```

The Conda environment is always named `LWCL`.

Before a paper-scale run:

1. Put data outside Git, then create a manifest under an ignored data directory.
2. Run subject-disjoint and environment-disjoint split audits.
3. Complete `remote_smoke.sh` including save/resume and test evaluation.
4. Confirm the Qwen model path and cache location.
5. Record the Git commit, resolved config, data-manifest checksum and GPU inventory.
6. Start with short checkpoint/evaluation intervals; increase them only after a successful resume test.

Paper route:

```bash
conda run --no-capture-output -n LWCL python -m lwcl.cli.train \
  --config configs/paper_qwen.yaml \
  --output-dir outputs/paper_qwen
```

Do not commit data, model weights, checkpoints, logs, credentials, `.env`, `.idea`, or local environment folders.
