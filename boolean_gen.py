import math
import numpy as np
from collections import Counter

# first try at LZ algorithm, many implementations, I think this is one of them?
def lempel_ziv_pass(word: str):
    # Should compress any string, but '#' not allowed (used as sentinel value)
    # lookup = alphabet
    lookup = set()
    idx = 0
    complexity = 0
    while True: # stop before
        length = 1
        idx_slice = slice(idx, idx + length)
        while word[idx_slice] in lookup:
            if idx + length < len(word):
                length += 1
                idx_slice = slice(idx, idx + length)
            else:
                return complexity + math.ceil(math.log(len(lookup) + 1, 2))

        complexity += math.ceil(math.log(len(lookup) + 1, 2))
        lookup.add(word[idx_slice])
        idx += length


def lempel_ziv(word: str):
    return lempel_ziv_pass(word)/2 + lempel_ziv_pass(''.join(reversed(word)))/2


def gen_boolean_functions(n: int):
    '''
    :param n: number of inputs
    :yield: all boolean functions of size n
    '''

    inputs = np.array([np.array([bool(int(j)) for j in format(i, f'0{n}b')]) for i in range(2**n)])

    for i in range(2**(2**n)):
        outputs = np.array([bool(int(j)) for j in format(i, f'0{2**n}b')])
        yield inputs, outputs


def boolean_to_key(vector):
    # Ensure the input is a boolean vector
    if not all(isinstance(x, (bool, np.bool_)) for x in vector):
        raise ValueError("The vector must be a boolean vector.")

    # Convert boolean vector to integer
    return int("".join('1' if x else '0' for x in vector), 2)


class SimpleDNN:
    def __init__(self, architecture):
        # Initialize weights for each layer based on the architecture
        self.weights = [np.random.randn(y, x) for x, y in zip(architecture[:-1], architecture[1:])]

    def forward(self, inputs):
        # Check if inputs have the right size
        if inputs.shape[1] != self.weights[0].shape[1]:
            raise ValueError("Input has incorrect size.")

        # Perform a forward pass
        activations = inputs
        for w in self.weights:
            activations = self.heaviside(np.dot(activations, w.T)) # Transpose weights to align for matrix multiplication
        return activations

    def sample(self, mu, sigma):
        # Sample new parameters for the network
        self.weights = [mu + sigma * np.random.randn(*w.shape) for w in self.weights]

    def mask(self, masks):
        # Apply a boolean mask to the parameters of the network
        for idx, w_mask in enumerate(masks):
            self.weights[idx] = self.weights[idx] * w_mask

    @staticmethod
    def sigmoid(z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def heaviside(x):
        return np.where(x < 0, 0, 1)


function_counter = Counter()
function_complexity = {}
n=4
inputs = np.array([np.array([bool(int(j)) for j in format(i, f'0{n}b')]) for i in range(2**n)])

net = SimpleDNN([n,n,1])
for i in range(1000000):
    net.sample(0,1)
    out = net.forward(inputs)
    function_counter.update([tuple(out.squeeze())])
    as_str = ''.join([str(b) for b in out.squeeze()])
    print(as_str)
    if tuple(out.squeeze()) not in function_complexity:
        function_complexity[tuple(out.squeeze())] = lempel_ziv(as_str)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x = []
y = []
for key in function_counter:
    x.append(function_complexity[key])
    y.append(function_counter[key])

plt.yscale('log')
plot = plt.scatter(x, y)
plt.show()
