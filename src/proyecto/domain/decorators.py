import functools
import time

"""
@author: bargof
@date: 15/04/2026
"""


def timer(func):
    """Mide el tiempo de ejecución de cualquier función."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = timer.perf_counter() - start
        print(f"[TIMER] {func.__name__} ejecutada en {elapsed: .4f}s")
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Reintenta una función si falla."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"  [RETRY] Intento {attempt}/{max_attempts} falló: {e}")
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)

        return wrapper

    return decorator
