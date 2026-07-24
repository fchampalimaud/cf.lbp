"""
task_loader.py — discovers and instantiates task plugins from tasks/.
"""

import os
import importlib
import inspect

from .task_base import BaseTask

_TASK_DIR = os.path.dirname(__file__)


def discover_tasks() -> list[str]:
    """Return sorted list of task module names (without .py, excluding internals)."""
    names = []
    for fname in os.listdir(_TASK_DIR):
        if fname.startswith('_') or not fname.endswith('.py'):
            continue
        if fname in ('task_base.py', 'task_loader.py'):
            continue
        names.append(fname[:-3])
    return sorted(names)


def load_task(name: str) -> BaseTask:
    """Import tasks/<name>.py and return an instance of the first BaseTask subclass."""
    mod = importlib.import_module(f'tasks.{name}')
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, BaseTask) and cls is not BaseTask:
            return cls()
    raise ValueError(f"No BaseTask subclass found in tasks/{name}.py")
