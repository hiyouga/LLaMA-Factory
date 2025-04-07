# Linear Probe CLIP

To run linear probe baselines, make sure that your current working directory is `lpclip/`.

Step 1: Extract Features using the CLIP Image Encoder

```bash
sh feat_extractor.sh
```

Step 2: Train few-shot linear probe

```bash
sh linear_probe.sh
```

We follow the instructions stated in the Appendix A3 (pp.38) of [the original CLIP paper](https://arxiv.org/pdf/2103.00020.pdf), with a careful hyperparameter sweep.

Note: please pull the latest Dassl (version >= `606a2c6`).
