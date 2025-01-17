# Restricted Boltzmann Machine (RBM) for Iris Dataset

This project implements a **Restricted Boltzmann Machine (RBM)** and applies it to the **Iris dataset**. The goal is to preprocess the Iris dataset, train an RBM model, and evaluate its performance in a binary classification task.

## Project Structure

- `data/`
  - `iris/`
    - `binary_preprocessed_iris.npz`: Preprocessed Iris dataset with binary transformation.
    - `preprocessed_iris.npz`: Regular preprocessed dataset with features and one-hot encoded labels.
    - `percentiles.npy`: Percentiles (33rd and 66th) for each feature in the Iris dataset.
- `model/`
  - `rbm.py`: Implementation of the Restricted Boltzmann Machine class.
  - `train.py`: Training logic for the RBM.
- `utils/`
  - Helper functions for preprocessing, activation, and RBM evaluation.

## Requirements

- Python 3.8+
- NumPy
- scikit-learn

Install dependencies using:

```bash
pip install numpy scikit-learn
```

## Usage
1. Preprocess the Iris Dataset
The Iris dataset is preprocessed by dividing the dataset into three categories based on petal length: small, medium, and large. Percentiles for each feature are also calculated and saved.

Run the following script to preprocess the dataset:
```bash
python preprocess.py
```

This will generate the following output files:
- `iris/preprocessed_iris.npz`: Regular preprocessed Iris dataset with one-hot encoded labels.
- `iris/percentiles.npy`: Percentile values for each feature.
- `iris/binary_preprocessed_iris.npz`: Binary transformed Iris dataset based on percentiles.

2. Train the RBM
The RBM can be trained on the preprocessed Iris dataset. Use the following script to train the model and save the weights:
```bash
python train_rbm.py
```

This will:
- Train the RBM on the preprocessed Iris dataset.
- Save the trained weights to `iris/rbm_weights.npz`.
- Calculate the accuracy of the trained RBM model.

3. Test the RBM
Once the RBM is trained, you can test its performance using the `test_rbm_init.py` and `test_rbm_trainer.py` scripts. These scripts initialize an RBM with random weights and calculate its accuracy on the Iris dataset.
```bash
python test_rbm_init.py
```
or
```bash
python test_rbm_trainer.py
```
This will output the accuracy of the RBM based on its predictions on the Iris dataset.
