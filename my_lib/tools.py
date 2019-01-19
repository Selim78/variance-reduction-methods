from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def Time():
    """
    Measures and yields elapsed time.
    Usage:

        with Time() as runtime:
            ...

        print("Computation time {0:0.3f} seconds".format(runtime()))
    """
    start = default_timer()
    elapsed = lambda: default_timer() - start
    yield elapsed
    end = default_timer()
    elapsed = lambda: end-start
