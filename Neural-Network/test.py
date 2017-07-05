import numpy as np
from neuralnet import *

# 28x28 pixel images are encoded as a length-784 numpy array
# Outputs are a length-10 numpy array corresponding to 10 decimal digits
layer_map = [784,30,10]


# Load data
num_samples = 6000
split_at = 5000
test_data = False
training_labels, training_images, test_labels, test_images = open_data.get_data(num_samples, training=True, test=test_data)

#np.random.shuffle(training_labels.T)
#np.random.shuffle(training_images.T)
#if test_data:
#    np.random.shuffle(test_labels.T)
#    np.random.shuffle(test_images.T)

test_labels = training_labels[split_at:]
test_images = (training_images[:,split_at:] / 255) - .5
training_labels = training_labels[:split_at]
training_images = (training_images[:,:split_at] / 255) - .5

# Initialize network
network = Network(layer_map)

def label_to_flag_vec(labels):
    # Reshape output scalars to the decimal encoding
    label_vecs = np.zeros((10,len(labels)))
    for i,val in enumerate(labels):
        label_vecs[int(val),i] = 1.0
    return label_vecs

training_label_vecs = label_to_flag_vec(training_labels)
test_label_vecs = label_to_flag_vec(test_labels)

epochs     = 1000
eta        = 2.5
batch_size = 10
tick       = 25
plotting = True

# Train network weights and labels on training data
#network.stochastic_gradient_descent(training_images, training_label_vecs, batch_size, epochs, eta, test_images, test_label_vecs, tick=tick, plot=plotting)

def test_and_log_params(epochs=1000,eta=1.0,batch_size=10,tick=10, log_file = "log.txt"):
    network.stochastic_gradient_descent(training_images, training_label_vecs, batch_size, epochs, eta, test_images, test_label_vecs, tick=tick, verbose=False)
    _, num_correct = network.cost_and_count(test_images, test_label_vecs)
    accuracy = num_correct / test_images.shape[1]

    out_string = "{train_size},{test_size},{epochs},{eta},{batch_size},{accuracy}\n".format(
            train_size=training_images.shape[1],
            test_size=test_images.shape[1],
            epochs=epochs, eta=eta, batch_size=batch_size, accuracy=accuracy)
    with open(log_file, "a") as f:
        f.write(out_string)

for learning_rate in np.arange(.25,4.25,.5):
    print(learning_rate)
    test_and_log_params(30,learning_rate,10,10)

