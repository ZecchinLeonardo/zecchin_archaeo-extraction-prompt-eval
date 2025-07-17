"""Test the caching mechanism for functions processing a batch."""

from typing import cast
from collections.abc import Iterator
from pathlib import Path

from joblib.memory import MemorizedFunc
from archaeo_super_prompt.utils.cache import (
    get_cache_dir_for,
    get_memory_for,
    identity_function,
    is_input_in_the_cache,
    manualy_cache_batch_processing,
    manually_cache_result,
    escape_expensive_run_when_cached,
)


@get_memory_for("external").cache(ignore=["output"])
def dummy_id_func(input: str, output: int | None = None):
    """Identity function for caching the outputs for each element of the batch."""
    return identity_function(input, output)


def test_cache():
    """Test the manual cache mechanism."""
    already_processed_input_set = set()

    def expensive_function(input: str):
        """Simulate a time-consuming function."""
        # check if the input has never been processed before
        assert input not in already_processed_input_set
        already_processed_input_set.add(input)
        return len(input) + 8

    inpts = ("hello", "bonsoir", "foo", "bar")
    for inpt in inpts:
        otpt = expensive_function(inpt)
        manually_cache_result(cast(MemorizedFunc, dummy_id_func), inpt, otpt)
        assert is_input_in_the_cache(cast(MemorizedFunc, dummy_id_func), inpt)
        cached_opt = dummy_id_func(inpt)
        assert cached_opt == otpt


def _dummy_id_func2(input: str, output: int | None = None):
    # we do not declare the memory cache as it is done by the batch processing
    return identity_function(input, output)


class NotHashable:
    """A not hashable object as it can be met in the usage of the cache module."""

    def __init__(self, val: str) -> None:
        """Init a not hasahble object."""
        self._val = val

    def hash(self):
        """Hash the value."""
        return self._val

    def get_val(self):
        """Get the value."""
        return self._val


def test_batch_cache():
    """Test the cache mechanism for batch processing.

    Test on a function which process a batch and return an iterable over the
    outputs for each input of the batch.
    """
    inpts = (NotHashable(v) for v in ("hello", "bonsoir", "foo", "bar"))
    inpts2 = (
        NotHashable(v)
        for v in ("hello", "au revoir", "bonsoir", "bar", "ciao")
    )

    def hash_fn(input: NotHashable) -> str:
        return input.hash()

    accumulated_inputs = set()

    def expensive_function(inputs: Iterator[NotHashable]):
        for inpt in inputs:
            h = inpt.hash()
            assert h not in accumulated_inputs
            accumulated_inputs.add(h)
            yield len(inpt.get_val()) + 9

    def escaper(iterab: Iterator[NotHashable]):
        return escape_expensive_run_when_cached(
            _dummy_id_func2,
            get_memory_for("external"),
            hash_fn,
            expensive_function,
            iterab,
        )

    iter1 = escaper(inpts)
    iter2 = escaper(inpts2)
    list(iter1)
    list(iter2)


def test_manual_batch_cache():
    """Test the manual caching for batch processing."""
    inputs1 = [2, 3, 8, 9]
    inputs2 = [2, 9, 7, 8, 7, 5]
    already_processed_input_set: set[int] = set()

    def expensive_function(batch: Iterator[int]):
        inpts = set(batch)
        for inpt in inpts:
            assert inpt not in already_processed_input_set
            already_processed_input_set.add(inpt)
            yield "*" * inpt

    def inpt_to_cache_path(inpt: int):
        return get_cache_dir_for("raw", "test") / f"test__{str(inpt)}.txt"

    def save_in_cache(otpt: str, inpt: Path):
        with inpt.open("w") as f:
            f.write(otpt)

    def load_from_cache(inpt: Path):
        with inpt.open("r") as f:
            return f.read()

    def lazy_process(inputs: list[int]):
        return manualy_cache_batch_processing(
            inpt_to_cache_path,
            save_in_cache,
            load_from_cache,
            expensive_function,
            iter(inputs),
        )

    lazy_process(inputs1)
    lazy_process(inputs2)
    for inpt in already_processed_input_set:
        file = inpt_to_cache_path(inpt)
        assert file.exists()
        with file.open("r") as f:
            assert f.read() == "*" * inpt
        file.unlink()
    assert len(list(get_cache_dir_for("raw", "test").iterdir())) == 0
