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
This will create a ```model/<data_type>/``` directory and a model file named ```train_model```. There are multiple arguments that need to be set, but the default setting is fine for VGG16. The settings used for other CKA models are commented in the code.

To train a LTH-based model:
```
python LTH_new.py -mode=train -data_type=cifar10 -model_type=vgg16 -seed=13 -learning_rate=0.1 -momentum=0.9 -prune_percentage=70 -num_epoch=200 -patience=3
```
This will create a ```model/<data_type>/``` directory and a model file named ```LTH_train_model```. There are multiple arguments that need to be set, but the default setting is fine for VGG16. The settings used for other LTH models are commented in the code.


To train a CL-based model:
```
python ckalth.py -mode=train -data_type=cifar10 -model_type=vgg16 -seed=13 -learning_rate=0.001 -momentum=0.9 -prune_percentage=70 -num_epoch=50 -patience=5
```
This will create a ```model/<data_type>/``` directory and a model file named ```CL_train_model```. There are multiple arguments that need to be set, but the default setting is fine for VGG16. The settings used for other CL models are commented in the code.

If a specific CKA, LTH, or CL model needs to be used, change class ```VGG16_N```  in ```src/model.py```. The model architectures and results are given [here](https://github.com/YHJYH/COMP0098_21049846/blob/main/model_architecture.md#list-of-architectures).

## Step 2: test the model
To test a CKA-based model:
```
python train.py -mode=train -data_type=cifar10 -model_type=vgg16 -seed=13 -learning_rate=0.001 -momentum=0.9 -num_epoch=50 -patience=5
```
This will create a ```output/<data_type>/``` directory and a model file named ```features.pt```. There are multiple arguments that need to be set, but the default setting is fine for VGG16. The settings used for other CKA models are commented in the code.

To test a LTH-based model:
```
python LTH_new.py -mode=train -data_type=cifar10 -model_type=vgg16 -seed=13 -learning_rate=0.1 -momentum=0.9 -prune_percentage=70 -num_epoch=200 -patience=3
```
This will create a ```output/<data_type>/``` directory and a model file named ```LTH_features.pt```. There are multiple arguments that need to be set, but the default setting is fine for VGG16. The settings used for other LTH models are commented in the code.


To test a CL-based model:
```
python ckalth.py -mode=train -data_type=cifar10 -model_type=vgg16 -seed=13 -learning_rate=0.001 -momentum=0.9 -prune_percentage=70 -num_epoch=50 -patience=5
```
This will create a ```output/<data_type>/``` directory and a model file named ```CL_features.pt```. There are multiple arguments that need to be set, but the default setting is fine for VGG16. The settings used for other CL models are commented in the code.

If a specific CKA, LTH, or CL model needs to be used, change class ```VGG16_N```  in ```src/model.py```. The model architectures and results are given [here](https://github.com/YHJYH/COMP0098_21049846/blob/main/model_architecture.md#list-of-architectures).
