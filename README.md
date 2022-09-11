# COMP0098_21049846
The master thesis project for COMP0098 MSc Computational Statistics and Machine Learning ProjectURL

This PyTorch code was used in the experiments of the thesis project:

The code is cleaned post-acceptance and may run into some errors. Although I did some quick check and saw the code ran fine, please create an issue if you encounter errors and I will try to fix them as soon as possible.

# Dataset
We used two different datasets: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009), and [CIFAR10.2](https://github.com/modestyachts/cifar-10.2) (Lu et al., 2020).

# Running the code
## Step 0: set up a virtual environment
```
conda env create -f environment.yml
conda activate thesis_1
```

## Step 1: train the model
To train a CKA-based model:
```
python train.py -mode=train -data_type=cifar10 -model_type=vgg16 -seed=13 -learning_rate=0.001 -momentum=0.9 -num_epoch=50 -patience=5
```
This will create a ```model/``` directory and a model file named ```train_model```. There are multiple arguments that need to be set, but the default setting is fine for VGG16. The settings used for other CKA models are commented in the code.
