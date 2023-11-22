import multiprocessing
import boolean_gen as bg
import numpy as np
import torch


if __name__ == '__main__':
    N = 7
    EPSILON = .01
    NUM_EPOCHS = 5000
    COUNTER_SAMPLES = 10
    NUM_TESTS = 1
    STOP_MAGNITUDE = 1
    HIDDEN = [40, 40]
    MODEL_ARGS = ([N, *HIDDEN, 1], torch.nn.functional.sigmoid, .01)

    num_processes = multiprocessing.cpu_count()//2
    from functools import partial

    partial_find_lambda = partial(bg.find_lambda, num_epochs=NUM_EPOCHS, num_tests=NUM_TESTS,
                                  stop_magnitude=STOP_MAGNITUDE, epsilon=EPSILON, model_args=MODEL_ARGS)

    inputs = np.array([np.array([bool(int(j)) for j in format(i, f'0{N}b')]) for i in range(2 ** N)])

    counter, complexity = bg.get_init_dist(N, HIDDEN, COUNTER_SAMPLES)
    sampled_functions = map(lambda x: (inputs, x), counter.keys())

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute the training task over multiple processes
        results = pool.map(partial_find_lambda, sampled_functions)

    import csv

    file_name = "PRELIM_DATA_TOY.csv"
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["function", "l0 complexity", "lz complexity"])
        for i, result in enumerate(results):
            writer.writerow([i, result, bg.lempel_ziv(format(i, f'0{2 ** N}b'))])