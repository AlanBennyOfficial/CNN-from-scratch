import cupy as np

class Softmax:
  # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def forward_batch(self, inputs):
        # inputs shape: (N, input_dim)
        totals = np.dot(inputs, self.weights) + self.biases
        exp = np.exp(totals)
        out = exp / np.sum(exp, axis=1, keepdims=True)
        self.last_totals_batch = totals
        self.last_input_batch = inputs
        return out

    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float
        '''
        t_exp = np.exp(self.last_totals)
        S = np.sum(t_exp)
        y = t_exp / S
        # Assume d_L_d_out has only one nonzero element; find its index
        index = int(np.nonzero(d_L_d_out)[0][0])
        one_hot = np.zeros_like(y)
        one_hot[index] = 1
        # Vectorized derivative of loss with respect to totals:
        dL_d_z = y - one_hot
        d_L_d_w = self.last_input[:, np.newaxis] @ dL_d_z[np.newaxis, :]
        d_L_d_b = dL_d_z
        d_L_d_inputs = self.weights @ dL_d_z
        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b
        return d_L_d_inputs.reshape(self.last_input_shape)