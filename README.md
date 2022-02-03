<img src="https://raw.githubusercontent.com/SAP/ml-model-watermarking/dev/docs/logo-dark.svg?sanitize=true#gh-dark-mode-only" alt="Hurl Logo" width="6000px"><img src="https://raw.githubusercontent.com/SAP/ml-model-watermarking/dev/docs/logo-light.svg?sanitize=true#gh-light-mode-only" alt="Hurl Logo" width="6000px">

<p align="center">
    <a href="https://api.reuse.software/info/github.com/SAP/ml-model-watermarking">
        <img alt="REUSE" src="https://api.reuse.software/badge/github.com/SAP/ml-model-watermarking">
    </a>
    <a href="https://github.com/SAP/ml-model-watermarking/blob/main/LICENSE">
        <img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
</p>


<h3 align="center">
    <p>Protect your machine learning models easily and securely with watermarking :key: </p>
</h3>

---

The concept of digital watermarking has been known for 30 years, mainly for image and audio contents. The goal is to insert a unique, hidden and non-removable signal in the original content, to be used as an identifier. If a thief steals a content, the original owner can still prove his/her ownership. ML Model Watermarking offers basic primitives for researchers and machine learning enthusiasts to watermark their models, without advanced knowledge of underlying concepts.

* :book: Watermark models on various tasks, such as **image classification** or **sentiment analysis**, with a compatibility with the main Machine Learning frameworks like [sklearn](https://github.com/scikit-learn/scikit-learn), [Pytorch](https://github.com/pytorch/pytorch) or the [HuggingFace library](https://github.com/huggingface/transformers).
* :triangular_flag_on_post: Detect if one of your models has been used without consent.
* :chart_with_upwards_trend: Integrate watermark in your pipeline, with a negligible accuracy loss.

## Installation


Simply run:

``` python
>>>  pip install .
```

## How to use it

ML Model Watermarking acts as a wrapper for your model, provoding a range of techniques for watermarking your model as well as ownership detection function. After the watermarking phase, you can retrieve your model and save the ownership information. 

``` python
>>> from mlmodelwatermarking.markface import TrainerWM

>>> trainer = TrainerWM(model=your_model)
>>> ownership = trainer.watermark()
>>> watermarked_model = trainer.model
```

Later, it is possible verify if a given model has been stolen based on the ownership information

``` python
>>> from mlmodelwatermarking.markface import TrainerWM
>>> from mlmodelwatermarking.verification import verify

>>> trainer = TrainerWM(model=suspect_model, ownership=ownership)
>>> trainer.verify()
{'is_stolen': True, 'score': 0.88, 'threshold': 0.66}
```


## References

The library implements several ideas presented in academic papers:

1. [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/conference/usenixsecurity18/presentation/adi)
2. [Embedding Watermarks into Deep Neural Networks](https://dl.acm.org/doi/abs/10.1145/3078971.3078974?casa_token=H5HTBeo2JDAAAAAA:P5P93MufED9DZZ5zAfqaaIJ5x2Y81t-HKfQLVPsRTC7XSaN7NaWUZA-1Wg2_F0ROIFCXzapYjsFs)
3. [Rethinking Stealthiness of Backdoor Attack against NLP Models](https://aclanthology.org/2021.acl-long.431.pdf)
4. [Yes We can: Watermarking Machine Learning Models beyond Classification](https://ieeexplore.ieee.org/document/9505220)

## Contributing

We invite your participation to the project through issues and pull requests. Please refer to the [Contributing guidelines](https://github.com/SAP/ml-model-watermarking/blob/main/CONTRIBUTING.md) for how to contribute.

## How to obtain support

You can open an [issue](https://github.com/SAP/ml-model-watermarking/issues).

## Licensing

Copyright 2020-21 SAP SE or an SAP affiliate company and ml-model-watermarking contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is [available via the REUSE tool](https://api.reuse.software/info/github.com/SAP/ml-model-watermarking).
