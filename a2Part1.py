import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from NeuralNetwork import Neural_Network


def encode_labels(labels):
    # encode 'Adelie' as 1, 'Gentoo' as 2, 'Chinstrap' as 3,...
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # don't worry about this
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, integer_encoded, onehot_encoder, onehot_encoded


if __name__ == '__main__':
    data = pd.read_csv('penguins307-train.csv')
    # the class label is last!
    labels = data.iloc[:, -1]
    # seperate the data from the labels
    instances = data.iloc[:, :-1]
    #scale features to [0,1] to improve training
    scaler = MinMaxScaler()
    instances = scaler.fit_transform(instances)
    # We can't use strings as labels directly in the network, so need to do some transformations
    label_encoder, integer_encoded, onehot_encoder, onehot_encoded = encode_labels(labels)
    # labels = onehot_encoded

    # Parameters. As per the handout.
    n_in = 4
    n_hidden = 2
    n_out = 3
    learning_rate = 0.2

    initial_hidden_layer_weights = np.array([[-0.28, -0.22], [0.08, 0.20], [-0.30, 0.32], [0.10, 0.01]])
    initial_output_layer_weights = np.array([[-0.29, 0.03, 0.21], [0.08, 0.13, -0.36]])

    nn = Neural_Network(n_in, n_hidden, n_out, initial_hidden_layer_weights, initial_output_layer_weights,
                        learning_rate)

    print('First instance has label {}, which is {} as an integer, and {} as a list of outputs.\n'.format(
        labels[0], integer_encoded[0], onehot_encoded[0]))

    # need to wrap it into a 2D array
    instance1_prediction = nn.predict([instances[0]])
    if instance1_prediction[0] is None:
        # This should never happen once you have implemented the feedforward.
        instance1_predicted_label = "???"
    else:
        instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction)
    print('Predicted label for the first instance is: {}\n'.format(instance1_predicted_label))

    # TODO: Perform a single backpropagation pass using the first instance only. (In other words, train with 1
    #  instance for 1 epoch!). Hint: you will need to first get the weights from a forward pass.
    hidden_layer_outputs,output_layer_outputs=nn.forward_pass(instances[0])
    delta_output_layer_weights, delta_hidden_layer_weights = nn.backward_propagate_error(instances[0],hidden_layer_outputs,output_layer_outputs,onehot_encoded[0])
    nn.update_weights(delta_output_layer_weights,delta_hidden_layer_weights)

    print('Weights after performing BP for first instance only:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    # TODO: Train for 100 epochs, on all instances.
    epochs=100
    nn.train(instances, onehot_encoded, epochs,integer_encoded)

    print('\nAfter training:')
    print('Hidden layer weights:\n', nn.hidden_layer_weights)
    print('Output layer weights:\n', nn.output_layer_weights)

    pd_data_ts = pd.read_csv('penguins307-test.csv')
    test_labels = pd_data_ts.iloc[:, -1]
    test_instances = pd_data_ts.iloc[:, :-1]
    #scale the test according to our training data.
    test_instances = scaler.transform(test_instances)

    # TODO: Compute and print the test accuracy
    test_label_encoder, test_integer_encoded, test_onehot_encoder, test_onehot_encoded = encode_labels(test_labels)
    trcount=0
    test_instance_prediction = nn.predict(test_instances)
    for i in range(len(test_instance_prediction)):
        if test_instance_prediction[i]==test_integer_encoded[i]:
            trcount+=1
    acc = str(round(trcount/len(test_instances)*100,2))+'%'
    print('acc = ', acc)