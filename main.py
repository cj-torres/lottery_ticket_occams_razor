import multiprocessing
import boolean_gen as bg
import torch


if __name__ == '__main__':
    N = 3
    EPSILON = .01
    NUM_EPOCHS = 5000
    NUM_TESTS = 1
    STOP_ACCURACY = .0005
    MODEL_ARGS = ([N, N, N, 1], torch.nn.functional.sigmoid, .01)

    num_processes = multiprocessing.cpu_count()
    from functools import partial

    partial_find_lambda = partial(bg.find_lambda, num_epochs=NUM_EPOCHS, num_tests=NUM_TESTS,
                                  stop_accuracy=STOP_ACCURACY, epsilon=EPSILON, model_args=MODEL_ARGS)

    data_generator = bg.gen_boolean_functions(N)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute the training task over multiple processes
        results = pool.map(partial_find_lambda, data_generator)

    import csv

    file_name = "PRELIM_DATA_TOY.csv"
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["function", "l0 complexity", "lz complexity"])
        for i, result in enumerate(results):
            writer.writerow([i, result, bg.lempel_ziv(format(i, f'0{2 ** N}b'))])