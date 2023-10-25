<img src="./magvit2.png" width="400px"></img>

## MagViT2 - Pytorch (wip)

Implementation of MagViT2 from <a href="https://arxiv.org/abs/2310.05737">Language Model Beats Diffusion - Tokenizer is Key to Visual Generation</a> in Pytorch. This currently holds SOTA for video generation / understanding.

The Lookup Free Quantizer proposed in the paper can be found in a <a href="https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/lookup_free_quantization.py">separate repository</a>. It should probably be explored for all other modalities, starting with <a href="https://github.com/lucidrains/audiolm-pytorch/commit/c748fcdb565964bc562277bd73fbeb2e5df0ffca">audio</a>

Please join <a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a> if you are interested in replicating the tokenizer proposed in this paper out in the open

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

## Install

```bash
$ pip install magvit2-pytorch
```

## Usage

```python
import torch
from magvit2_pytorch.magvit2_pytorch import VideoTokenizer

tokenizer = VideoTokenizer(
    image_size = 256,
    init_dim = 64,
    layers = (
        'residual',
        ('compress_space', 128),
        'residual',
        'residual',
        'attend_space',
        ('compress_time', 256),
        'attend_time'
    )
)

# get a ton of videos

videos = torch.randn(2, 3, 16 + 1, 256, 256) # (batch, channels, time, height, width)

# course it through the autoencoder

total_loss, loss_breakdown = tokenizer(videos, return_loss = True)
total_loss.backward()

# after much training above, you can get the tokenized codes

tokenizer.eval()
codes = tokenizer(videos, return_codes = True)

```

## Todo

- [ ] Magvit2 Tokenizer
    - [x] add adversarial loss
    - [x] implement the blurpool for antialiasing in discriminator
    - [x] LFQ should be able to pass loss breakdown (commitment and entropy), and forwarded to the return of the tokenizer
    - [x] add conditioning for encoder decoder with residual modulatable conv 3d
    - [ ] add adaptive rmsnorm
    - [ ] add trainer and manage discriminator training
    - [ ] completely generalize to multiple discriminators at different time scales (taking inspiration of multi-resolution discriminators from soundstream)
    - [ ] add attention
        - [ ] use axial rotary embeddings for spatial
    - [ ] add an optional autoregressive loss at some penultimate layer of the decoder - check literature to see if anyone else has done this unification of transformer decoder + tokenizer in one architecture

- [ ] MaskGit

## Citations

```bibtex
@misc{yu2023language,
    title   = {Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation}, 
    author  = {Lijun Yu and José Lezama and Nitesh B. Gundavarapu and Luca Versari and Kihyuk Sohn and David Minnen and Yong Cheng and Agrim Gupta and Xiuye Gu and Alexander G. Hauptmann and Boqing Gong and Ming-Hsuan Yang and Irfan Essa and David A. Ross and Lu Jiang},
    year    = {2023},
    eprint  = {2310.05737},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@article{Zhang2021TokenST,
    title   = {Token Shift Transformer for Video Classification},
    author  = {Hao Zhang and Y. Hao and Chong-Wah Ngo},
    journal = {Proceedings of the 29th ACM International Conference on Multimedia},
    year    = {2021}
}
```

```bibtex
@article{Shleifer2021NormFormerIT,
    title     = {NormFormer: Improved Transformer Pretraining with Extra Normalization},
    author    = {Sam Shleifer and Jason Weston and Myle Ott},
    journal   = {ArXiv},
    year      = {2021},
    volume    = {abs/2110.09456},
    url       = {https://api.semanticscholar.org/CorpusID:239016890}
}
```

```bibtex
@inproceedings{ElNouby2021XCiTCI,
    title   = {XCiT: Cross-Covariance Image Transformers},
    author  = {Alaaeldin El-Nouby and Hugo Touvron and Mathilde Caron and Piotr Bojanowski and Matthijs Douze and Armand Joulin and Ivan Laptev and Natalia Neverova and Gabriel Synnaeve and Jakob Verbeek and Herv{\'e} J{\'e}gou},
    booktitle = {Neural Information Processing Systems},
    year    = {2021},
    url     = {https://api.semanticscholar.org/CorpusID:235458262}
}
```
