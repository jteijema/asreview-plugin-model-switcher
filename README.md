# A new ASReview mode

This plugin is currently in development.

## Functions
This new model switches classifier at a set point.


## Getting started

Run install.sh or install the new classifier with:

```bash
pip install .
```

or

```bash
pip install git@github.com:JTeijema/ASReview-Model_Switcher.git
```


## Usage

The new classifier `SVM_LSTM` is defined in
[`asreviewcontrib/models/SVM_LSTM.py`](asreviewcontrib/models/SVM_LSTM.py) 
and can be used in a simulation.

The new classifier `NB_NN2L` is defined in
[`asreviewcontrib/models/NB_NN2L.py`](asreviewcontrib/models/NB_NN2L.py) 
and can be used in a simulation.

```bash
asreview simulate example_data_file.csv -m SVM_LSTM
```

```bash
asreview simulate example_data_file.csv -m NB_NN2L
```

## License
Apache-2.0 License 
