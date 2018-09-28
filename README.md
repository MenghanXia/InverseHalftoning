## Deep Inverse Halftoning

We run this code under [TensorFlow](https://www.tensorflow.org) 1.6 on Ubuntu16.04 with python pakage IPL installed.

### Network Architecture

TensorFlow Implementation of our paper ["Deep Inverse Halftoning via Progressively Residual Learning"](https://menghanxia.github.io) accepted to ACCV 2018.

<div align="center">
	<img src="img/network.jpg" width="90%">
</div>

### Results

<div align="center">
	<img src="img/example.jpg" width="90%">
</div>

### Preparation

- You can run exisitng halftone algorithm to generate halftone version of your continuous-tone grayscale or color images, working as training pairs.
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

- Start evaluation. (pretrained model is provided in the chechpoints folder)
```python
line 238: parser.add_argument('--mode', type=str, default='test', help='train, test')
```
```bash
python3 main.py 
```