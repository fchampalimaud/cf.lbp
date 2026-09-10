"""
app_version.py — reads the 2D simulator's release version.

The VERSION file (simulation/2d/VERSION) is bumped automatically by the
post-commit hook in scripts/git-hooks/ (installed via scripts/install-hooks.sh)
whenever a commit whose message starts with 'fix:' or 'feat:'/'add:' touches
files under simulation/2d/ — see rules/git_workflow.md.
"""

import os

_VERSION_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'VERSION')


def get_app_version() -> str:
    try:
        with open(_VERSION_FILE, encoding='utf-8') as f:
            return f.read().strip() or '0.0.0'
    except OSError:
        return '0.0.0'
