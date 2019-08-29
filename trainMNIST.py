from Neural_Network import*

import mnist_loader
import pickle
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

network = Neural_Network([784, 2, 10])
accuracies = network.train(training_data, epochs = 2, mini_batch_size = 10, learning_rate = 0.1, regularization = 5, test_data=test_data, save_name = False)
plt.plot(accuracies)
plt.title("Accuracy Growth")
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.show()
