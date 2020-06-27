# PyTorch Reimplementation - MisGAN: Learning From Incomplete Data With Generative Adversarial Networks

![ConvMisGAN](test/epoch300.png "Conv-MisGAN")

## Requirements
This code was tested on:

- Python 3.6
- PyTorch 1.5.0
- Google Colab

## Run
Source code is located in the `src` directory. The reimplementation only focuses on MNIST data. Jupyter notebooks in the `test` directory can also be run in isolation.

In the `src` directory,

Conv-MisGAN on MNIST:
```
python conv_misgan.py
```

FC-MisGAN on MNIST:
```
python fc_misgan.py
```

## References
- [Original Paper](https://openreview.net/forum?id=S1lDV3RcKm) (OpenReview)
- [DCGAN](https://arxiv.org/abs/1511.06434) (arXiv)
- [WGAN with Gradient Penalty](https://arxiv.org/abs/1704.00028) (arXiv)
