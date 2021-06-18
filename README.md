# A new ASReview model
This model switches between 2 models during runtime. It can be useful for when later stages of data classification require different models.

This plugin contains a base model and examples for a switching model, Contains 1 base switcher and 2 implementations.


## Getting started

Run install.sh or install the new classifier with:

```bash
pip install .
```

or

```bash
python -m pip install git+https://github.com/JTeijema/ASReview-Model_Switcher.git
```


## Usage

The new base switcher is defined in
[`asreviewcontrib/models/model_switcher.py`](asreviewcontrib/models/model_switcher.py).

The new classifier `SVM_LSTM` is defined in
[`asreviewcontrib/models/SVM_LSTM.py`](asreviewcontrib/models/SVM_LSTM.py) 
and can be used in a simulation.

The new classifier `NB_NN2L` is defined in
[`asreviewcontrib/models/NB_NN2L.py`](asreviewcontrib/models/NB_NN2L.py) 
and can be used in a simulation.

The models can be used like this:
```bash
asreview simulate example_data_file.csv -m SVM_LSTM -e doc2vec
```

```bash
asreview simulate example_data_file.csv -m NB_NN2L -e tfidf
```

### Switch point
Currently the switch point is set manually in model_switcher.py, named ``switchpoint``.

## License
Apache-2.0 License 
