# A new ASReview mode

This plugin is currently in development.

## Functions
This new model switches classifier at a set point.


## Getting started

Install the new classifier with

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

```bash
asreview simulate example_data_file.csv -m SVM_LSTM
```

## License
Apache-2.0 License 
