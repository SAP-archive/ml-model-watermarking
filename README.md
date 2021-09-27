# ML Model Watermarking

[![REUSE status](https://api.reuse.software/badge/github.com/SAP/ml-model-watermarking)](https://api.reuse.software/info/github.com/SAP/ml-model-watermarking)

ML Model Watermarking is a library for watermarking machine learning models, compatible with the main Machine Learning frameworks like [sklearn](https://github.com/scikit-learn/scikit-learn) and [Pytorch](https://github.com/pytorch/pytorch).

## Goals

The concept of digital watermarking has been known for 30 years, mainly for image and audio contents. The goal is to insert a unique, hidden and non-removable signal in the original content, to be used as an identifier. If a thief steals a content, the original owner can still prove his/her ownership. Indeed, given the efficiency of watermarking to ensure the protection of intellectual property of its users, researchers considered to adapt watermarking to protect machine learning models. ML Model Watermarking offers basic primitives to watermarking ML models, without advanced knowledge of underlying concepts.

## Installation

``` python
pip install .
```

## How to use it

The library is split into two modules: MarkLearn for [sklearn](https://github.com/scikit-learn/scikit-learn) and MarkTorch for [Pytorch](https://github.com/pytorch/pytorch). if you want more information, you can check the tests folder.

### MarkLearn

For sklearn models, specify the model instance to be watermarked as the first argument, then use ```wm_model``` like a normal sklearn model (with ```fit```, ```predict```, etc...). For example: 

``` python
from sklearn import datasets
from mlmodelwatermarking.marklearn.marklearn import MarkLearn


wm_model = MarkLearn(SGDClassifier(), encryption=False, metric='accuracy')
wine = datasets.load_wine()
X, y = wine.data, wine.target
ownership = wm_model.fit(X_train, y_train)
```
Information about the watermark is stored in the dictionnary ```ownership```

### MarkTorch

For Pytorch models, after specifying basic elements (architecture, optimizer, loss function, etc.), indicate parameters related to watermark.

``` python
from mlmodelwatermarking.marktorch.marktorch import MarkTorch

# WATERMARKED
model = MNIST()
trainset, valset, testset = load_MNIST()

trainer = MarkTorch(
                model=model,
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                criterion=nn.NLLLoss(),
                trainset=trainset,
                valset=valset,
                testset=testset,
                nbr_classes=10)

ownership = trainer.train(epochs=5)
```

## Models/Tasks supported

### Sklearn
- [x] [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [x] [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
- [x] [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [x] [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

### Pytorch
- [x] MNIST
- [ ] CIFAR10 

## References

The library implements several ideas presented in academic papers:

- [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/conference/usenixsecurity18/presentation/adi)
- [Embedding Watermarks into Deep Neural Networks](https://dl.acm.org/doi/abs/10.1145/3078971.3078974?casa_token=H5HTBeo2JDAAAAAA:P5P93MufED9DZZ5zAfqaaIJ5x2Y81t-HKfQLVPsRTC7XSaN7NaWUZA-1Wg2_F0ROIFCXzapYjsFs)
- [Entangled Watermarks as a Defense against Model Extraction](https://www.usenix.org/conference/usenixsecurity21/presentation/jia)

## Contributing

We invite your participation to the project through issues and pull requests. Please refer to the [Contributing guidelines](https://github.com/SAP/ml-model-watermarking/blob/main/CONTRIBUTING.md) for how to contribute.

## How to obtain support

You can open an [issue](https://github.com/SAP/ml-model-watermarking/issues).

## Licensing

Copyright 2020-21 SAP SE or an SAP affiliate company and ml-model-watermarking contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is [available via the REUSE tool](https://api.reuse.software/info/github.com/SAP/ml-model-watermarking).
