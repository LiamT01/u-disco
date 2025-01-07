from typing import Callable, TypeVar

cache_dict = {}

T = TypeVar('T')


def register_cache(key: str, callback: Callable[[], T], verbose=False) -> T:
    if key not in cache_dict:
        cache_dict[key] = callback()
    elif verbose:
        print(f"Cache hit for key: {key}")
    return cache_dict[key]


def get_cache(key: str) -> any:
    return cache_dict[key]
