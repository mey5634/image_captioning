### What

A webservice accepting a JPEG and returning a caption:
- forked [DeepRNN/image_captioning](https://github.com/DeepRNN/image_captioning)
    for a pre-trained model.
- [Flask](http://flask.pocoo.org/) endpoint

### Prerequisites

1. Install [pyenv](https://github.com/pyenv/pyenv-installer) to set up
    an isolated Python interpreter and workspace
2. Install Python2.7 with `pyenv`: `pyenv install 2.7.15`
3. Set up a workspace:
```shell
pyenv virtualenv 2.7.15 image_captioning
git clone -b hackathon --single-branch https://github.com/mey5634/image_captioning.git
```
4. `cd` into the repo, run `echo "image_captioning" > .python-version` to
    activate/deactivate the virtualenv on switching in/out of the directory
5. pull in dependencies with `pip install -r requirements.txt`
6. Pull in an `nltk` dependency: `python -c "import nltk; nltk.download(\"punkt\")"`
7. on MacOS only: `echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc`
8. Download:
- [pretrained model](https://app.box.com/s/xuigzzaqfbpnf76t295h109ey9po5t8p) to `./models/289999.npy`
- `wget` http://images.cocodataset.org/annotations/annotations_trainval2014.zip 
    and extract `captions_train2014.json` under `./train` and `captions_val2014.json` 
    under `./val`
9. test that everything works:
```shell
python main.py --phase=test \
    --model_file='./models/289999.npy' \
    --beam_size=3
```

### Quickstart

TODO: describe starting service & api

---

## Original README

### Introduction
This neural system for image captioning is roughly based on the paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (ICML2015). The input is an image, and the output is a sentence describing the content of the image. It uses a convolutional neural network to extract visual features from the image, and uses a LSTM recurrent neural network to decode these features into a sentence. A soft attention mechanism is incorporated to improve the quality of the caption. This project is implemented using the Tensorflow library, and allows end-to-end training of both CNN and RNN parts.

### Prerequisites
* **Tensorflow** ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](https://scipy.org/install.html))
* **OpenCV** ([instructions](https://pypi.python.org/pypi/opencv-python))
* **Natural Language Toolkit (NLTK)** ([instructions](http://www.nltk.org/install.html))
* **Pandas** ([instructions](https://scipy.org/install.html))
* **Matplotlib** ([instructions](https://scipy.org/install.html))
* **tqdm** ([instructions](https://pypi.python.org/pypi/tqdm))

### Usage
* **Preparation:** Download the COCO train2014 and val2014 data [here](http://cocodataset.org/#download). Put the COCO train2014 images in the folder `train/images`, and put the file `captions_train2014.json` in the folder `train`. Similarly, put the COCO val2014 images in the folder `val/images`, and put the file `captions_val2014.json` in the folder `val`. Furthermore, download the pretrained VGG16 net [here](https://app.box.com/s/idt5khauxsamcg3y69jz13w6sc6122ph) or ResNet50 net [here](https://app.box.com/s/17vthb1zl0zeh340m4gaw0luuf2vscne) if you want to use it to initialize the CNN part.

* **Training:**
To train a model using the COCO train2014 data, first setup various parameters in the file `config.py` and then run a command like this:
```shell
python main.py --phase=train \
    --load_cnn \
    --cnn_model_file='./vgg16_no_fc.npy'\
    [--train_cnn]    
```
Turn on `--train_cnn` if you want to jointly train the CNN and RNN parts. Otherwise, only the RNN part is trained. The checkpoints will be saved in the folder `models`. If you want to resume the training from a checkpoint, run a command like this:
```shell
python main.py --phase=train \
    --load \
    --model_file='./models/xxxxxx.npy'\
    [--train_cnn]
```
To monitor the progress of training, run the following command:
```shell
tensorboard --logdir='./summary/'
```

* **Evaluation:**
To evaluate a trained model using the COCO val2014 data, run a command like this:
```shell
python main.py --phase=eval \
    --model_file='./models/xxxxxx.npy' \
    --beam_size=3
```
The result will be shown in stdout. Furthermore, the generated captions will be saved in the file `val/results.json`.

* **Inference:**
You can use the trained model to generate captions for any JPEG images! Put such images in the folder `test/images`, and run a command like this:
```shell
python main.py --phase=test \
    --model_file='./models/xxxxxx.npy' \
    --beam_size=3
```
The generated captions will be saved in the folder `test/results`.

### Results
A pretrained model with default configuration can be downloaded [here](https://app.box.com/s/xuigzzaqfbpnf76t295h109ey9po5t8p). This model was trained solely on the COCO train2014 data. It achieves the following BLEU scores on the COCO val2014 data (with `beam size=3`):
* **BLEU-1 = 70.3%**
* **BLEU-2 = 53.6%**
* **BLEU-3 = 39.8%**
* **BLEU-4 = 29.5%**

Here are some captions generated by this model:
![examples](examples/examples.jpg)

### References
* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. ICML 2015.
* [The original implementation in Theano](https://github.com/kelvinxu/arctic-captions)
* [An earlier implementation in Tensorflow](https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow)
* [Microsoft COCO dataset](http://mscoco.org/)
