import math
import numpy as np
import torch
import mdl_layers_louizos as mll

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


class SimpleNumpyDNN:
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


class MDLDNN(torch.nn.Module):
    def __init__(self, architecture, activation, lam, wd=0):
        super(MDLDNN, self).__init__()
        # Initialize weights for each layer based on the architecture
        self.weights = torch.nn.ModuleList([mll.RegLinear(x, y, lam, wd)
                                            for x, y in zip(architecture[:-1], architecture[1:])])
        self.activation = activation
        self.lam = lam
        self.wd = wd

    def forward(self, inputs):
        # batch first input
        # Check if inputs have the right size
        if inputs.shape[1] != self.weights[0].weight.param.shape[0]:
            raise ValueError("Input has incorrect size.")

        # Perform a forward pass
        activations = inputs
        for w in self.weights:
            activations = self.activation(w(activations, block_sample=True)) # Transpose weights to align for matrix multiplication
        return activations

    def reset_parameters(self):
        # Sample new parameters for the network
        for w in self.weights:
            w.reset_parameters()

    def constrain_parameters(self):
        for w in self.weights:
            w.constrain_parameters()

    def set_lam(self, lam):
        self.lam = lam
        for w in self.weights:
            w.set_lam(lam)

    def set_wd(self, wd):
        self.wd = wd
        # Apply a boolean mask to the parameters of the network
        for w in self.weights:
            w.set_wd(wd)

    def regularization(self):
        return torch.stack([w.regularization() for w in self.weights]).sum()

    def count_l0(self):
        return sum([w.count_l0() for w in self.weights])

    def count_l2(self):
        return sum([w.count_l2() for w in self.weights])


def train_model(data_tuple, num_epochs, lam, *args):
    # Unpack the data tuple
    x, y = data_tuple
    SAMPLES_PER_STEP = 4

    # Convert boolean tensors to long type if necessary
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    # Initialize the model with the provided arguments
    model = MDLDNN(*args)
    model.set_lam(lam)

    # Define the loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

    #breakpoint()
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(x)
        ce_loss = criterion(outputs.squeeze(), y)
        l0 = model.regularization()
        loss = ce_loss + l0

        loss.backward()
        optimizer.step()

        #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {ce_loss.item():.4f}, L0: {l0.item():.4f}')

    return criterion(model(x).squeeze(), y).item(), model.count_l0().item()




def find_lambda(boolean_function, num_epochs, num_tests, stop_accuracy, epsilon, model_args):
    '''
    :param boolean_function: boolean function to test
    :param num_epochs: number of epochs to train the function on
    :param num_tests: how many trials to perform for each lambda
    :param stop_accuracy: how close
    :param epsilon: allowable deviation from loss
    :param model_args: args to parameterize MDLDNN model
    :return:
    '''

    def lambda_test(num_tests, epsilon, boolean_function, num_epochs, lam, *args):
        l0_results = []
        for i in range(num_tests):
            loss, l0 = train_model(boolean_function, num_epochs, lam, *args)
            if loss < epsilon:
                l0_results.append(l0)

        if l0_results:
            best_l0 = min(l0_results)
        else:
            best_l0 = None

        return best_l0

    lower_bound = 0
    upper_bound = 1
    best_l0_overall = float('inf')

    while upper_bound - lower_bound > stop_accuracy:
        lam = (upper_bound + lower_bound) / 2

        best_l0_from_test = lambda_test(num_tests, epsilon, boolean_function, num_epochs, lam, *model_args)

        if best_l0_from_test:
            print(f'Functional l0 found with {best_l0_from_test:.3f} parameters.')
            lower_bound = lam
            if best_l0_from_test < best_l0_overall:
                best_l0_overall = best_l0_from_test
        else:
            upper_bound = lam

        print(f'Target lambda between {lower_bound:.5f} and {upper_bound:.5f}, best solution contains {best_l0_overall:.4f} parameters.')

    return best_l0_overall


def get_init_dist(n):
    function_counter = Counter()
    function_complexity = {}
    inputs = np.array([np.array([bool(int(j)) for j in format(i, f'0{n}b')]) for i in range(2**n)])

    net = SimpleNumpyDNN([n, n, n, 1])
    for i in range(1000000):
        net.sample(0,1)
        out = net.forward(inputs)
        function_counter.update([tuple(out.squeeze())])
        as_str = ''.join([str(b) for b in out.squeeze()])
        print(as_str)
        if tuple(out.squeeze()) not in function_complexity:
            function_complexity[tuple(out.squeeze())] = lempel_ziv(as_str)

    return function_counter, function_complexity


# def parallel_train(n):
#     EPSILON = .01
#     NUM_EPOCHS = 5000
#     NUM_TESTS = 5
#     STOP_ACCURACY = .0025
#     MODEL_ARGS = ([n,n,n,1], torch.nn.functional.sigmoid, .01)
#
#     num_processes = multiprocessing.cpu_count()
#     from functools import partial
#     partial_find_lambda = partial(find_lambda, num_epochs = NUM_EPOCHS, num_tests = NUM_TESTS,
#                                   stop_accuracy = STOP_ACCURACY, epsilon = EPSILON, model_args = MODEL_ARGS)
#
#     data_generator = gen_boolean_functions(n)
#
#     with multiprocessing.Pool(processes=num_processes) as pool:
#         # Distribute the training task over multiple processes
#         results = pool.map(partial_find_lambda, data_generator)
#
#     import csv
#     file_name = "PRELIM_DATA.csv"
#     with open(file_name, "w") as file:
#         writer = csv.writer(file)
#         writer.writerow(["function", "l0 complexity", "lz complexity"])
#         for i, result in enumerate(results):
#             writer.writerow([1, result, lempel_ziv(format(i, f'0{2**n}b'))])







# function_counter = Counter()
# function_complexity = {}
# n=4
# inputs = np.array([np.array([bool(int(j)) for j in format(i, f'0{n}b')]) for i in range(2**n)])
#
# net = SimpleNumpyDNN([n,n,1])
# for i in range(1000000):
#     net.sample(0,1)
#     out = net.forward(inputs)
#     function_counter.update([tuple(out.squeeze())])
#     as_str = ''.join([str(b) for b in out.squeeze()])
#     print(as_str)
#     if tuple(out.squeeze()) not in function_complexity:
#         function_complexity[tuple(out.squeeze())] = lempel_ziv(as_str)
#
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# x = []
# y = []
# for key in function_counter:
#     x.append(function_complexity[key])
#     y.append(function_counter[key])
#
# plt.yscale('log')
# plot = plt.scatter(x, y)
# plt.show()
