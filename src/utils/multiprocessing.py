import os
from multiprocessing import Pool
from typing import TypeVar, Callable, List, Iterable

T_FUNC_ARG = TypeVar('T_FUNC_ARG')
T_FUNC_RET = TypeVar('T_FUNC_RET')


def get_num_workers() -> int:
    num_cpus: int = os.cpu_count() if os.cpu_count() is not None else 1
    return int(os.environ.get("SLURM_CPUS_PER_TASK", num_cpus))


def map_tasks(
        func: Callable[[T_FUNC_ARG], T_FUNC_RET],
        args: List[T_FUNC_ARG],
        num_workers: int = None,
) -> Iterable[T_FUNC_RET]:
    """
    Run the given function with the given arguments in parallel using multiprocessing.Pool
    :param func: Function to run, must take a single argument (does not require unpacking)
    :param args: List of arguments, each element is passed to the function separately
    :param num_workers: Number of workers to use, defaults to number of CPUs
    :return: List of results from running the function multiple times
    """
    if num_workers is None:
        num_workers = get_num_workers()

    with Pool(num_workers) as pool:
        results = pool.map(func, args)
    return results
