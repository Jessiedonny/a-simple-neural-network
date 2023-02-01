import numpy as np


class Neural_Network:
    # Initialize the network
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights, output_layer_weights, learning_rate, hidden_node_bias,output_node_bias):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights

        self.learning_rate = learning_rate
        self.hidden_node_bias = hidden_node_bias
        self.output_node_bias = output_node_bias

    # Calculate neuron activation for an input
    def sigmoid(self, input):
        #output = np.NaN  # TODO!
        output=1/(1 + np.exp(-input))
        return output

    # Feed forward pass input to a network output
    def forward_pass(self, inputs):
        hidden_layer_outputs = []
        for i in range(self.num_hidden):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum=0
            for j in range(self.num_inputs):
                weighted_sum+=inputs[j]*self.hidden_layer_weights[j][i]
            output = self.sigmoid(weighted_sum+self.hidden_node_bias[i])
            hidden_layer_outputs.append(output)

        output_layer_outputs = []
        for i in range(self.num_outputs):
            # TODO! Calculate the weighted sum, and then compute the final output.
            weighted_sum = 0
            for j in range(self.num_hidden):
                weighted_sum+=hidden_layer_outputs[j]*self.output_layer_weights[j][i]
            output = self.sigmoid(weighted_sum+self.output_node_bias[i])
            output_layer_outputs.append(output)

        return hidden_layer_outputs, output_layer_outputs

    # Backpropagate error and store in neurons
    def backward_propagate_error(self, inputs, hidden_layer_outputs, output_layer_outputs, desired_outputs):

        output_layer_betas = np.zeros(self.num_outputs)
        # TODO! Calculate output layer betas.
        for i in range(self.num_outputs):
            output_layer_betas[i]=desired_outputs[i]-output_layer_outputs[i]
        
        hidden_layer_betas = np.zeros(self.num_hidden)
        # TODO! Calculate hidden layer betas.
        for j in range(self.num_hidden):
            for k in range(self.num_outputs):
                hidden_layer_betas[j]+=self.output_layer_weights[j][k]*output_layer_outputs[k]*(1-output_layer_outputs[k])*output_layer_betas[k]

        # This is a HxO array (H hidden nodes, O outputs)
        delta_output_layer_weights = np.zeros((self.num_hidden, self.num_outputs))
        # TODO! Calculate output layer weight changes.
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                delta_output_layer_weights[i][j]=self.learning_rate*hidden_layer_outputs[i]*output_layer_outputs[j]*(1-output_layer_outputs[j])*output_layer_betas[j]

        # This is a IxH array (I inputs, H hidden nodes)
        delta_hidden_layer_weights = np.zeros((self.num_inputs, self.num_hidden))
        # TODO! Calculate hidden layer weight changes.
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                delta_hidden_layer_weights[i][j]=self.learning_rate*inputs[i]*hidden_layer_outputs[j]*(1-hidden_layer_outputs[j])*hidden_layer_betas[j]

        #bias
        delta_hidden_node_bias = np.zeros(self.num_hidden)
        #Calculate hidden node bias changes
        for i in range(self.num_hidden):
            delta_hidden_node_bias[i]=self.learning_rate*hidden_layer_outputs[i]*(1-hidden_layer_outputs[i])*hidden_layer_betas[i]
        
        delta_output_node_bias = np.zeros(self.num_outputs)
        #Calculate hidden node bias changes
        for i in range(self.num_outputs):
             delta_output_node_bias[i]=self.learning_rate*output_layer_outputs[i]*(1-output_layer_outputs[i])*output_layer_betas[i]
        
       
        # Return the weights we calculated, so they can be used to update all the weights.
        return delta_output_layer_weights, delta_hidden_layer_weights,delta_hidden_node_bias,delta_output_node_bias

    def update_weights(self, delta_output_layer_weights, delta_hidden_layer_weights,delta_hidden_node_bias,delta_output_node_bias):
        # TODO! Update the weights.
        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                self.hidden_layer_weights[i][j]+=delta_hidden_layer_weights[i][j]
        for i in range(self.num_hidden):
            for j in range(self.num_outputs):
                self.output_layer_weights[i][j]+=delta_output_layer_weights[i][j]
        for i in range(self.num_hidden):
            self.hidden_node_bias[i]+=delta_hidden_node_bias[i]
        for i in range(self.num_outputs):
            self.output_node_bias[i]+=delta_output_node_bias[i]
        #print('Placeholder')

    def train(self, instances, desired_outputs, epochs,integer_encoded):
        for epoch in range(epochs):
            print('epoch = ', epoch)
            predictions = []
            for i, instance in enumerate(instances):
                hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
                delta_output_layer_weights, delta_hidden_layer_weights,delta_hidden_node_bias,delta_output_node_bias = self.backward_propagate_error(
                    instance, hidden_layer_outputs, output_layer_outputs, desired_outputs[i])
                predicted_class = output_layer_outputs.index(max(output_layer_outputs))   # TODO!
                predictions.append(predicted_class)

                # We use online learning, i.e. update the weights after every instance.
                self.update_weights(delta_output_layer_weights, delta_hidden_layer_weights,delta_hidden_node_bias,delta_output_node_bias)

            # Print new weights
            print('Hidden layer weights \n', self.hidden_layer_weights)
            print('Output layer weights  \n', self.output_layer_weights)
            print('Hidden nodes bias \n', self.hidden_node_bias)
            print('Output nodes bias \n', self.output_node_bias)

            # TODO: Print accuracy achieved over this epoch
            rcount=0
            instance_prediction = self.predict(instances)
            for i in range(len(instance_prediction)):
                if instance_prediction[i]==integer_encoded[i]:
                    rcount+=1
            acc = str(round(rcount/len(instances)*100,2))+'%'
            print('acc = ', acc)

    def predict(self, instances):
        predictions = []
        for instance in instances:
            hidden_layer_outputs, output_layer_outputs = self.forward_pass(instance)
            predicted_class = output_layer_outputs.index(max(output_layer_outputs))  # TODO! Should be 0, 1, or 2.
            predictions.append(predicted_class)
        return predictions