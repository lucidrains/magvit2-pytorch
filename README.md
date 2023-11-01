<img src="./magvit2.png" width="400px"></img>

## MagViT2 - Pytorch

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
from magvit2_pytorch import (
    VideoTokenizer,
    VideoTokenizerTrainer
)

tokenizer = VideoTokenizer(
    image_size = 128,
    init_dim = 64,
    layers = (
        'residual',
        ('compress_space', 128),
        'attend_space',
        'linear_attend_space',
        'residual',
        'residual',
        ('consecutive_residual', 3),
        ('compress_time', 256),
        'attend_time',
    )
)

trainer = VideoTokenizerTrainer(
    tokenizer,
    dataset_folder = '/path/to/a/lot/of/media',     # folder of either videos or images, depending on setting below
    dataset_type = 'videos',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = 16,
    grad_accum_every = 4,
    num_train_steps = 1_000_000
)

trainer.train()

# after a lot of training ...
# can use the EMA of the tokenizer

ema_tokenizer = trainer.ema_tokenizer

# mock video

video = torch.randn(1, 3, 17, 32, 32)

# tokenizing video to discrete codes

codes = ema_tokenizer.tokenize(video) # (1, 9, 16, 16) <- time and space downsampled by 2x. flatten token ids for (non)-autoregressive training

# sanity check

decoded_video = ema_tokenizer.decode_from_code_indices(codes)

assert torch.allclose(
    decoded_video,
    ema_tokenizer(video, return_recon = True)
)
```

## Todo

- [ ] Magvit2 Tokenizer
    - [x] add adversarial loss
    - [x] implement the blurpool for antialiasing in discriminator
    - [x] LFQ should be able to pass loss breakdown (commitment and entropy), and forwarded to the return of the tokenizer
    - [x] add conditioning for encoder decoder with residual modulatable conv 3d
    - [x] `decode_from_codebook_indices` should be able to accept flattened ids and reshape to correct feature map dimensions and decode back to video
    - [x] add trainer and manage discriminator training
    - [x] completely generalize to multiple discriminators at different time scales (taking inspiration of multi-resolution discriminators from soundstream)
        - [x] complete multiscale discriminator losses
        - [x] auto-manage multiscale discriminator optimizers
        - [ ] helper functions for crafting multi-resolution temporal discriminators (picking random consecutive frames)
    - [ ] add adaptive rmsnorm
    - [ ] add attention
        - [ ] use axial rotary embeddings for spatial
    - [ ] add an optional autoregressive loss at some penultimate layer of the decoder - check literature to see if anyone else has done this unification of transformer decoder + tokenizer in one architecture
- [ ] Improvise a <a href="https://arxiv.org/abs/2203.01941">RQ Video Transformer</a>, as residual LFQ actually makes sense now

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
