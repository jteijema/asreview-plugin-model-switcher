# ASReview Model Switcher
This model switches between 2 models during runtime. It can be useful for when later stages of data classification require different models.

This plugin contains a base model and examples for a switching model, Contains a base switcher and its implementation.

The current switching model uses SVM and a convolutional neural network, and needs doc2vec as feature extractor.


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
[`asreviewcontrib/models/svm_nn.py`](asreviewcontrib/models/svm_cnn.py) 
and can be used in a simulation.

The models can be used like this:
```bash
asreview simulate example_data_file.csv -m svm_cnn -e doc2vec
```

### Switch point
Currently the switch point is set manually in model_switcher.py, named ``switchpoint``.

## License
Apache-2.0 License 


## other
Currently, a new convolutional neural network is implemented [`asreviewcontrib/models/cnn.py`](asreviewcontrib/models/cnn.py) , usable with:
```bash
asreview simulate example_data_file.csv -m power_cnn -e doc2vec
```
