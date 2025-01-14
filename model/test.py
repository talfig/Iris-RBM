from model.rbm import RBM
from data.loader import loader
from utils import initialize_weights
from model.train import RBMTrainer


def test():
    # Example usage
    a, b, J = initialize_weights()
    rbm = RBM(a, b, J)

    file_path = "../data/iris/binary_preprocessed_iris.npz"
    features, labels = loader(file_path)
    sample, label = features[0], labels[0]
    print(f'Prediction: {rbm.contrastive_divergence(sample)[0][:3]}, true label: {label}')


def test_rbm_trainer():
    # Initialize the RBMTrainer
    trainer = RBMTrainer()
    file_path = "../data/iris/binary_preprocessed_iris.npz"

    # Train the RBM
    print("Training the RBM...")
    a, b, J = trainer.train_rbm(file_path)

    # Create an RBM instance with the trained parameters
    rbm = RBM(a, b, J)

    # Load the dataset
    features, labels = loader(file_path)

    # Test predictions on the entire dataset
    correct_count = 0
    for i, (sample, label) in enumerate(zip(features, labels)):
        prediction = rbm.contrastive_divergence(sample)[0][:3]  # Get the prediction for the first 3 spots
        is_correct = (prediction == label[:3])  # Compare prediction with true label
        correct_count += is_correct

        print(f"Sample {i + 1}: Prediction {prediction}, True Label {label[:3]}, Correct: {is_correct}")

    # Calculate and display the accuracy
    accuracy = correct_count / len(features) * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")


# Call the test function
test_rbm_trainer()
