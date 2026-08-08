from pathlib import Path


def find_project_root() -> Path:
    """Return the project root from either the root or notebooks folder."""

    current_path = Path.cwd()

    if current_path.name == "notebooks":
        return current_path.parent

    return current_path