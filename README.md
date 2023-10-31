
# ShiftTolerant-LPIPS

### Update

[2023 Aug] Added ST-LPIPS to [PyTorch Toolbox for Image Quality Assessment](https://github.com/chaofengc/IQA-PyTorch).  

[2023 May] In the ST-LPIPS work (available in this repository), we developed a perceptual similarity metric that remains robust even in the presence of imperceptible pixel shifts. However, stronger corruptions can be generated via adversarial attacks. Consequently, _**in a separate study**_, we conducted a systematic investigation of the robustness of both traditional and learned perceptual similarity metrics against imperceptible adversarial perturbations. Our findings reveal that all metrics are susceptible to such attacks. For details, please consider reading our study on '[Attacking Perceptual Similarity Metrics](https://github.com/abhijay9/attacking_perceptual_similarity_metrics)' (TMLR'23 $\textcolor{red}{\text{Featured Certification}}$).

## Shift-tolerant Perceptual Similarity Metric

[Abhijay Ghildyal](https://abhijay9.github.io/), [Feng Liu](http://web.cecs.pdx.edu/~fliu/). In ECCV, 2022. [[Arxiv]](https://arxiv.org/abs/2207.13686), [[PyPI]](https://pypi.org/project/stlpips-pytorch/), [[video]](https://www.youtube.com/watch?v=F6C5VQJGIrM)

<img src="https://abhijay9.github.io/images/stlpips_teaser.gif" width=300>

### Quick start

`pip install stlpips_pytorch`

```python
from stlpips_pytorch import stlpips
from stlpips_pytorch import utils

path0 = "<dir>/ShiftTolerant-LPIPS/imgs/ex_p0.png"
path1 = "<dir>/ShiftTolerant-LPIPS/imgs/ex_ref.png"

img0 = utils.im2tensor(utils.load_image(path0))
img1 = utils.im2tensor(utils.load_image(path1))

stlpips_metric = stlpips.LPIPS(net="alex", variant="shift_tolerant")
# or for the vgg variant use `stlpips.LPIPS(net="vgg", variant="shift_tolerant")`

stlpips_distance = stlpips_metric(img0,img1).item()
```

or, please clone this repo and run 
```python 
python lpips_2imgs.py
```

### Training

```python
nohup python -u ./train.py --from_scratch --train_trunk \
    --use_gpu --gpu_ids 0 \
    --net alex --variant vanilla --name alex_vanilla \
    > logs/train_alex_vanilla.out &

nohup python -u ./train.py --from_scratch --train_trunk \
    --use_gpu --gpu_ids 1 \
    --net alex --variant shift_tolerant --name alex_shift_tolerant \
    > logs/train_alex_shift_tolerant.out &

nohup python -u ./train.py --from_scratch --train_trunk \
    --use_gpu --gpu_ids 2 \
    --net vgg --variant vanilla --name vgg_vanilla \
    > logs/train_vgg_vanilla.out &

nohup python -u ./train.py --from_scratch --train_trunk \
    --use_gpu --gpu_ids 3 \
    --net vgg --variant shift_tolerant --name vgg_shift_tolerant \
    > logs/train_vgg_shift_tolerant.out &
```

### Testing

Please download the original BAPPS dataset using this [script (here)](https://github.com/richzhang/PerceptualSimilarity/blob/master/scripts/download_dataset.sh). Then, update path to the dataset in [global_config.json](https://github.com/abhijay9/ShiftTolerant-LPIPS/tree/main/util/config).

To reproduce the results in the paper run the following:
```python

# bash n_pixel_shift_study/test_scripts/test.sh <net> <variant> <gpu_id> <img_resize> <batch_size>

# AlexNet Vanilla
nohup bash n_pixel_shift_study/test_scripts/test.sh alex vanilla 0 64 50 > logs/eval_alex_vanilla.out &

# AlexNet Shift-tolerant
nohup bash n_pixel_shift_study/test_scripts/test.sh alex shift_tolerant 1 64 50 > logs/eval_alex_shift_tolerant.out &

# Vgg Vanilla
nohup bash n_pixel_shift_study/test_scripts/test.sh vgg vanilla 2 64 50 > logs/eval_vgg_vanilla.out &

# Vgg Shift-tolerant
nohup bash n_pixel_shift_study/test_scripts/test.sh vgg shift_tolerant 3 64 50 > logs/eval_vgg_shift_tolerant.out &
```

**Note:** To train and test our models in this paper, we used Image.BICUBIC. The results are similar when other resizing methods are used. Please feel free to switch back to bilinear as used in the original LPIPS work (here).

### Other Evaluations

For other evaluations refer to [./n_pixel_shift_study/](https://github.com/abhijay9/ShiftTolerant-LPIPS/tree/main/n_pixel_shift_study).

## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{ghildyal2022stlpips,
  title={Shift-tolerant Perceptual Similarity Metric},
  author={Abhijay Ghildyal and Feng Liu},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

## Acknowledgements
This repository borrows from [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [Anti-aliasedCNNs](https://github.com/adobe/antialiased-cnns), and [CNNsWithoutBorders](https://github.com/oskyhn/CNNs-Without-Borders). We thank the authors of these repositories for their incredible work and inspiration.
