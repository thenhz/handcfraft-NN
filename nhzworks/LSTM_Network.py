import numpy as np
from neurons.LSTM import LSTM

class RecurrentNeuralNetwork:
    # input (word), expected output (next word), num of words (num of recurrences), array expected outputs, learning rate
    def __init__(self, n_input, n_output, n_steps, eo, learning_rate):
        # initial input (first word)
        self.x = np.zeros(n_input)
        # input size
        self.n_inputs = n_input
        # expected output (next word)
        self.y = np.zeros(n_output)
        # output size
        self.n_output = n_output
        # weight matrix for interpreting results from LSTM cell (num words x num words matrix)
        self.w = np.random.normal(size=(n_output, n_output), scale=5.0).astype(np.float32)
        # matrix used in RMSprop
        self.G = np.zeros_like(self.w)
        # length of the recurrent network - number of recurrences i.e num of words
        self.n_steps = n_steps
        # learning rate
        self.learning_rate = learning_rate
        # array for storing inputs
        self.inputs = np.zeros((n_steps + 1, n_input))
        # array for storing cell states
        self.cell_states = np.zeros((n_steps + 1, n_output))
        # array for storing outputs
        self.outputs = np.zeros((n_steps + 1, n_output))
        # array for storing hidden states
        self.hidden_states = np.zeros((n_steps + 1, n_output))
        # forget gate
        self.forget_gates = np.zeros((n_steps + 1, n_output))
        # input gate
        self.input_gates = np.zeros((n_steps + 1, n_output))
        # cell state
        self.cell_states = np.zeros((n_steps + 1, n_output))
        # output gate
        self.output_gates = np.zeros((n_steps + 1, n_output))
        # array of expected output values
        self.eo = np.vstack((np.zeros(eo.shape[0]), eo.T))
        # declare LSTM cell (input, output, amount of recurrence, learning rate)
        self.LSTM = LSTM(n_input, n_output, n_steps, learning_rate)

    # activation function. simple nonlinearity, convert nums into probabilities between 0 and 1
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # the derivative of the sigmoid function. used to compute gradients for backpropagation
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

        # lets apply a series of matrix operations to our input (curr word) to compute a predicted output (next word)

    def forwardProp(self):
        for i in range(1, self.n_steps + 1):
            self.LSTM.x = np.hstack((self.hidden_states[i - 1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            # store computed cell state
            self.cell_states[i] = cs
            self.hidden_states[i] = hs
            self.forget_gates[i] = f
            self.input_gates[i] = inp
            self.cell_states[i] = c
            self.output_gates[i] = o
            self.outputs[i] = self.sigmoid(np.dot(self.w, hs))
            self.x = self.eo[i - 1]
        return self.outputs

    def backProp(self):
        # update our weight matrices (Both in our Recurrent network, as well as the weight matrices inside LSTM cell)
        # init an empty error value
        totalError = 0
        # initialize matrices for gradient updates
        # First, these are RNN level gradients
        # cell state
        dfcs = np.zeros(self.n_output)
        # hidden state,
        dfhs = np.zeros(self.n_output)
        # weight matrix
        tu = np.zeros((self.n_output, self.n_output))
        # Next, these are LSTM level gradients
        # forget gate
        tfu = np.zeros((self.n_output, self.n_inputs + self.n_output))
        # input gate
        tiu = np.zeros((self.n_output, self.n_inputs + self.n_output))
        # cell unit
        tcu = np.zeros((self.n_output, self.n_inputs + self.n_output))
        # output gate
        tou = np.zeros((self.n_output, self.n_inputs + self.n_output))
        # loop backwards through recurrences
        for i in range(self.n_steps, -1, -1):
            # error = calculatedOutput - expectedOutput
            error = self.outputs[i] - self.eo[i]
            # calculate update for weight matrix
            # (error * derivative of the output) * hidden state
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.outputs[i])), np.atleast_2d(self.hidden_states[i]).T)
            # Time to propagate error back to exit of LSTM cell
            # 1. error * RNN weight matrix
            error = np.dot(error, self.w)
            # 2. set input values of LSTM cell for recurrence i (horizontal stack of arrays, hidden + input)
            self.LSTM.x = np.hstack((self.hidden_states[i - 1], self.inputs[i]))
            # 3. set cell state of LSTM cell for recurrence i (pre-updates)
            self.LSTM.cs = self.cell_states[i]
            # Finally, call the LSTM cell's backprop, retreive gradient updates
            # gradient updates for forget, input, cell unit, and output gates + cell states & hiddens states
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.cell_states[i - 1], self.forget_gates[i], self.input_gates[i], self.cell_states[i],
                                                            self.output_gates[i], dfcs, dfhs)
            # calculate total error (not necesarry, used to measure training progress)
            totalError += np.sum(error)
            # accumulate all gradient updates
            # forget gate
            tfu += fu
            # input gate
            tiu += iu
            # cell state
            tcu += cu
            # output gate
            tou += ou
        # update LSTM matrices with average of accumulated gradient updates
        self.LSTM.update(tfu / self.n_steps, tiu / self.n_steps, tcu / self.n_steps, tou / self.n_steps)
        # update weight matrix with average of accumulated gradient updates
        self.update(tu / self.n_steps)
        # return total error of this iteration
        return totalError

    def update(self, u):
        # vanilla implementation of RMSprop
        self.G = 0.9 * self.G + 0.1 * u ** 2
        self.w -= self.learning_rate / np.sqrt(self.G + 1e-8) * u
        return

    # this is where we generate some sample text after having fully trained our model
    # i.e error is below some threshold
    def sample(self):
        # loop through recurrences - start at 1 so the 0th entry of all arrays will be an array of 0's
        for i in range(1, self.n_steps + 1):
            # set input for LSTM cell, combination of input (previous output) and previous hidden state
            self.LSTM.x = np.hstack((self.hidden_states[i - 1], self.x))
            # run forward prop on the LSTM cell, retrieve cell state and hidden state
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            # store input as vector
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.inputs[i] = self.x  # Use np.argmax?
            # store cell states
            self.cell_states[i] = cs
            # store hidden state
            self.hidden_states[i] = hs
            # forget gate
            self.forget_gates[i] = f
            # input gate
            self.input_gates[i] = inp
            # cell state
            self.cell_states[i] = c
            # output gate
            self.output_gates[i] = o
            # calculate output by multiplying hidden state with weight matrix
            self.outputs[i] = self.sigmoid(np.dot(self.w, hs))
            # compute new input
            maxI = np.argmax(self.outputs[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        # return all outputs
        return self.outputs

    def train(self, epochs, iterations):
        for ii in range(1,epochs):
            for i in range(1, iterations):
                # compute predicted next word
                self.forwardProp()
                # update all our weights using our error
                error = self.backProp()
                # once our error/loss is small enough
                print("Error on iteration ", i, ": ", error)

    def test(self):
        # we can finally define a seed word
        seed = np.zeros_like(self.x)
        maxI = np.argmax(np.random.random(RNN.x.shape))
        seed[maxI] = 1
        RNN.x = seed
        # and predict some new text!
        output = RNN.sample()
        print(output)
        # write it all to disk
        ExportText(output, data)
        print("Done Writing")