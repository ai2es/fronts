import logging
import time

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def benchmark(dataset: tf.data.Dataset, num_epochs: int = 2):
    """Benchmark the execution time of iterating through a TensorFlow Dataset.

    Args:
        dataset: A TensorFlow Dataset to benchmark.
        num_epochs: The number of epochs to iterate through the dataset.
    """
    start_time = time.perf_counter()
    for _ in range(num_epochs):
        for _ in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)
