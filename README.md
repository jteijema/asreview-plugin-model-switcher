# ASReview Model Switcher
This repository contains a plugin for [ASReview](https://github.com/asreview) ![logo](https://raw.githubusercontent.com/asreview/asreview-artwork/e2e6e5ea58a22077b116b9c3d2a15bc3fea585c7/SVGicons/IconELAS/ELASeyes24px24px.svg "ASReview"). This plugin adds a model that switches between two models during runtime. It can be useful for when later stages of data classification require different models.

This plugin contains a base model and an implementation of a switching model.

The current implemented switching model uses SVM and a convolutional neural network, and needs doc2vec as feature extractor.


## Getting started

Run install.sh or install the new classifier with:

```bash
pip install .
```

or

```bash
python -m pip install git+https://github.com/JTeijema/asreview-plugin-model-switcher.git
```


## Usage

The new base switcher is defined in
[`asreviewcontrib/models/model_switcher.py`](asreviewcontrib/models/model_switcher.py).

The new classifier `svm_cnn` is defined in
[`asreviewcontrib/models/svm_cnn.py`](asreviewcontrib/models/svm_cnn.py) 
and can be used in a simulation.

The models can be used like this:
```bash
asreview simulate benchmark:van_de_Schoot_2017 -m svm_cnn -e doc2vec -s test.h5
```

### Switch point
Currently the switch point is set manually in model_switcher.py, named ``switchpoint``. It is set to 100 iterations.

## License
Apache-2.0 License 


## other
A new convolutional neural network is implemented [`asreviewcontrib/models/cnn.py`](asreviewcontrib/models/cnn.py) , usable with:
```bash
asreview simulate benchmark:van_de_Schoot_2017 -m power_cnn -e doc2vec
```
