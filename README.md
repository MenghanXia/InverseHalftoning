## Deep Inverse Halftoning

We run this code under [TensorFlow](https://www.tensorflow.org) 1.6 on Ubuntu16.04 with python pakage IPL installed.

### Network Architecture

TensorFlow Implementation of our paper ["Deep Inverse Halftoning via Progressively Residual Learning"](http://menghanxia.github.io/papers/2018_Inverse_Halftone_accv.pdf) accepted to ACCV 2018.

<div align="center">
	<img src="img/network.jpg" width="90%">
</div>

### Results

<div align="center">
	<img src="img/examples.jpg" width="90%">
</div>

### Preparation

- You can run exisitng halftone algorithm (*Foyd-Steinberg Error diffusion on 8-bit grayscale image is used in our pretrained model*) to generate halftone version of your continuous-tone grayscale or color images, working as training pairs.
- The patch size is set to 256x256 in the `model.py` (you may change it to any other size as you like).
- Download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).

### Run
- Set your image folders and hyperparameters in `main.py`.

- Start training.
```python
line238: parser.add_argument('--mode', type=str, default='train', help='train, test')
```
```bash
python3 main.py
```

- Start evaluation. (acess [pretrained model](https://drive.google.com/open?id=11wXkRgM-D55biKUPGz7EiSt1TR-1q2iA ))
```python
line 238: parser.add_argument('--mode', type=str, default='test', help='train, test')
```
```bash
python3 main.py 
```

### Copyright and License
You are granted with the [license](./LICENSE.txt) for both academic and commercial usages.

### Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@inproceedings{XiaYXZX17,
  author    = {Menghan Xia and Jian Yao and Renping Xie and Mi Zhang and Jinsheng Xiao},
  title     = {Color Consistency Correction Based on Remapping Optimization for Image Stitching},
  booktitle = {{IEEE} International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2017}
}
```
