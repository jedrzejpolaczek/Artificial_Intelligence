class ArtificialNeuron:
    """
    Class is one of many ways of implementation the model of McCulloch-Pitts threshold neuron (1943).
    Idea is to sum up all inputs multiply by weights add bias and give it as output.
    Formula y = f(s) where s = bias + dot(inputs, weights)

    Based on 'Stephen Marsland: Machine Learning : an algorithmic perspective' and
    https://en.wikipedia.org/wiki/Artificial_neuron.
    """
    def __init__(self, inputs: list, weights: list, bias: float):
        """
        McCulloch-Pitts artificial neuron gets on start inputs, weights and bias.
        Inputs may be different in "lifetime", weights will change in process of learning of artificial neuron and
        we can change the bias to make corrections if we see the learning process is not going right.

        :param inputs: vector of data we need to process. It pictures linear part of the neuron.
        :param weights: vector of the values with which we change the inputs value.
                        Weights is our way to "teach" neuron how to treats inputs.
        :param bias: value we can use bias for moving activation threshold along axis X.
                     A little way how to give us a bit of control on what happening.
        """
        self.inputs = inputs
        self.weights = weights
        self.bias = bias

        # In this implementation in output we store activation value.
        self.output = None

    def _dot(self):
        """
        Calculate dot product (mathematically speaking).
        The method is implemented here only for educational purposes.
        It can be replaced by method dot from library numpy.

        :return list: dot product for inputs and weights.
        """
        return sum([self.inputs[i] * self.weights[i] for i in range(len(self.weights))])

    def _sum_of_products(self):
        """
        Calculate sum of the product of inputs and weights with addition of bias.
        Mathematically speaking we are counting s in equation y = f(s)

        :return int: sum of products of inputs and theirs weights.
        """
        return self._dot() + self.bias

    def _activation_function(self):
        """
        Calculate value of activation function (activation value).
        Activation function also is named as transfer function or threshold function.
        Activation function can be linear or not.
        Mathematically speaking we are counting y in equation y = f(s)
        """
        if self._sum_of_products() >= 1:
            self.output = 1
        else:
            self.output = 0

    def forward(self):
        """
        Counting artificial network output. Output is value of activation function.
        """
        self._activation_function()


if __name__ == "__main__":
    # In this example we implementing activation function as some kind logical OR
    # (if sum is greater or equal 1 is true).

    neuron_input = [0, 0, 1, 1, 0]
    neuron_weights = [10, -9, 7, 5]
    neuron_bias = 0  # Value 0 is mathematically neutral in the equation.
    McC_P_threshold_neuron = ArtificialNeuron(neuron_input, neuron_weights, neuron_bias)
    McC_P_threshold_neuron.forward()
    print(McC_P_threshold_neuron.output)

    neuron_input = [1, 0, 10, 0, 0]
    neuron_weights = [-10, -9, -7, -5]
    neuron_bias = 0  # Value 0 is mathematically neutral in the equation.
    McC_P_threshold_neuron = ArtificialNeuron(neuron_input, neuron_weights, neuron_bias)
    McC_P_threshold_neuron.forward()
    print(McC_P_threshold_neuron.output)

    """
    NOTE: 
    
    How we can see one neuron can give us only one output so it will not be able to handle complex data.
    """