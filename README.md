# ML Model Watermarking

ML Model Watermarking is a library for watermarking machine learning models, compatible with the main Machine Learning frameworks like [sklearn](https://github.com/scikit-learn/scikit-learn) and [Pytorch](https://github.com/pytorch/pytorch).

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
- [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/conference/usenixsecurity18/presentation/adi)
- [Embedding Watermarks into Deep Neural Networks](https://dl.acm.org/doi/abs/10.1145/3078971.3078974?casa_token=H5HTBeo2JDAAAAAA:P5P93MufED9DZZ5zAfqaaIJ5x2Y81t-HKfQLVPsRTC7XSaN7NaWUZA-1Wg2_F0ROIFCXzapYjsFs)
- [Entangled Watermarks as a Defense against Model Extraction](https://www.usenix.org/conference/usenixsecurity21/presentation/jia)
