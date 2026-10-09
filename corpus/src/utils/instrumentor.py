import time
import json
import threading
import functools


class ProfileResult:
    def __init__(self, name: str, start: int, end: int, thread_id: int, args: dict | None = None):
        self.name = name
        self.start = start
        self.end = end
        self.thread_id = thread_id
        self.args = args


class InstrumentationSession:
    def __init__(self, name: str):
        self.name = name


class Instrumentor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.m_current_session = None
                cls._instance.m_output_stream = None
                cls._instance.m_profile_count = 0
        return cls._instance

    def begin_session(self, name: str, filepath: str = "profile.json"):
        if self.m_output_stream and not self.m_output_stream.closed:
            self.end_session()  # Ensure previous session is closed if not already

        self.m_output_stream = open(filepath, "w")
        self._write_header()
        self.m_current_session = InstrumentationSession(name)

    def end_session(self):
        if self.m_output_stream:
            self._write_footer()
            self.m_output_stream.close()
            self.m_output_stream = None
        self.m_current_session = None
        self.m_profile_count = 0

    def has_session(self) -> bool:
        return self.m_output_stream is not None and not self.m_output_stream.closed

    def write_profile(self, result: ProfileResult):
        if not self.m_output_stream:
            return

        if self.m_profile_count > 0:
            self.m_output_stream.write(",")

        name = result.name.replace('"', "'")  # Replace double quotes with single quotes

        profile_data = {
            "cat": "function",
            "dur": (result.end - result.start),
            "name": name,
            "ph": "X",
            "pid": 0,
            "tid": result.thread_id,
            "ts": result.start,
        }
        if result.args:
            profile_data["args"] = result.args
        self.m_output_stream.write(json.dumps(profile_data))
        self.m_output_stream.flush()
        self.m_profile_count += 1

    def _write_header(self):
        if self.m_output_stream:
            self.m_output_stream.write('{"otherData": {},"traceEvents":[')
            self.m_output_stream.flush()

    def _write_footer(self):
        if self.m_output_stream:
            self.m_output_stream.write("]}")
            self.m_output_stream.flush()

    @classmethod
    def get(cls):
        return cls()
    
    def __del__(self):
        self.end_session()


class InstrumentationTimer:
    def __init__(self, name: str, args: dict | None = None):
        self.m_name = name
        self.m_args = args
        self.m_stopped = False
        self.m_start_timepoint = self._get_current_microseconds()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.m_stopped:
            self.stop()

    def stop(self):
        end_timepoint = self._get_current_microseconds()

        thread_id = threading.current_thread().ident
        Instrumentor.get().write_profile(
            ProfileResult(self.m_name, self.m_start_timepoint, end_timepoint, thread_id, self.m_args)
        )

        self.m_stopped = True

    def _get_current_microseconds(self) -> int:
        return int(time.time_ns() / 1000)  # Convert nanoseconds to microseconds


# --- Decorator for profiling functions ---
def profile_function(func):
    """
    A decorator to automatically profile the execution of a function.
    The profile name will be the function's name.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with InstrumentationTimer(func.__name__):
            result = func(*args, **kwargs)
        return result

    return wrapper


# You can also create a decorator that allows specifying a custom name
def profile_scope(name: str):
    """
    A decorator factory to profile a function with a custom name.
    Usage: @profile_scope("My Custom Function Name")
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with InstrumentationTimer(name):
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator
