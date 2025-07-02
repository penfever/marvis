"""
System utilities for MARVIS.
"""

import signal
from contextlib import contextmanager


@contextmanager
def timeout_context(seconds):
    """Context manager for setting a timeout on a block of code."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler and a alarm for the timeout
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler and cancel the alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)