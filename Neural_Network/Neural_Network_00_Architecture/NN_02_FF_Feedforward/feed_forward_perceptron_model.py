import numpy as np
from NN_01_Rosenblatt_perceptron.perceptron_model import ArtificialNeuron


def training_algorithm(input_layer, output_layer, epochs, expected_output, acceptance_criteria):
    """
    Change neuron weights based on backpropagation described by Paul Werbos in 1974.
    Each implementation may be a little different but the general idea is
    in each epoch we want to change weights to be a little closer to the expected output.

    :param input_layer: contains input layer. Layer contains input artificial neurons.
    :param output_layer: contains output layer. Layer contains output artificial neuron.
    :param epochs: number of times teaching algorithm will repeat itself.
    :param expected_output: expected output we will use to calculate if we need to update a weights.
    :param acceptance_criteria: according to this value we will decide if weights need to be update.
    """
    for _ in range(epochs):
        # Step 2 and 3: Forward propagation
        input_layer.forward()
        output_layer.inputs = input_layer.output
        output_layer.forward()

        # Step 4: Backpropagation
        error = expected_output - output_layer.output
        corrected_predicted_output = error * (output_layer.output * (1 - output_layer.output))

        error_hidden_layer = corrected_predicted_output.dot(output_layer.weights.T)
        d_hidden_layer = error_hidden_layer * (input_layer.output * (1 - input_layer.output))

        # Step 5: Updating Weights and Biases
        if(error[0] > acceptance_criteria) or \
                (error[1] > acceptance_criteria) or \
                (error[2] > acceptance_criteria) or \
                (error[3] > acceptance_criteria):
            output_layer.weights += input_layer.output.T.dot(corrected_predicted_output)
            output_layer.bias += np.sum(corrected_predicted_output)
            input_layer.weights += inputs.T.dot(d_hidden_layer)
            input_layer.bias += np.sum(d_hidden_layer)


if __name__ == "__main__":
    # Input datasets
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])

    print("Initial inputs: ", end='')
    print(*inputs)
    print("Expected output: ", end='')
    print(*expected_output)

    # Step 1: Generate random weights for artificial neurons and creating perceptron from 3 artificial neurons
    cells_input_bias = np.random.uniform(size=(1, 2))
    cells_input_weights = np.random.uniform(size=(2, 2))
    input_layer = ArtificialNeuron(inputs, cells_input_weights, cells_input_bias)

    output_cell_weights = np.random.uniform(size=(2, 1))
    output_cell_bias = np.random.uniform(size=(1, 1))
    output_layer = ArtificialNeuron(None, output_cell_weights, output_cell_bias)

    print("\n---Initial summary---")
    # One artificial neuron == cell
    print("Initial input layer weights for cell 1:  ", end='')
    print(*input_layer.weights[0])
    print("Initial input layer biases  for cell 1:  ", end='')
    print(*input_layer.bias)
    print("Initial input layer weights for cell 2:  ", end='')
    print(*input_layer.weights[1])
    print("Initial input layer biases  for cell 2:  ", end='')
    print(*input_layer.bias)
    print("Initial output layer weights:            ", end='')
    print(*output_layer.weights)
    print("Initial output layer biases:             ", end='')
    print(*output_layer.bias)

    # Steps 2-5:
    training_algorithm(input_layer=input_layer,
                       output_layer=output_layer,
                       epochs=10000,
                       expected_output=expected_output,
                       acceptance_criteria=0.01)

    print("\n---Final summary---")
    print("Final input layer weights for cell 1:    ", end='')
    print(*input_layer.weights[0])
    print("Final input layer biases  for cell 1:    ", end='')
    print(*input_layer.bias)
    print("Final input layer weights for cell 2:    ", end='')
    print(*input_layer.weights[1])
    print("Final input layer biases  for cell 2:    ", end='')
    print(*input_layer.bias)
    print("Final output layer weights:              ", end='')
    print(*output_layer.weights)
    print("Final output layer biases:               ", end='')
    print(*output_layer.bias)

    print("\n---Results---")
    print("After 10 000 epochs we get, ")
    print("output from neural network: ", end='')
    print(output_layer.output[0], output_layer.output[1], output_layer.output[2], output_layer.output[3], sep='\t\t')
    print("compare to expected output: ", end='')
    print(expected_output[0], expected_output[1], expected_output[2], expected_output[3], sep='\t\t\t\t\t')
