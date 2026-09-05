# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import string
import tempfile

from pathlib import Path


def generate_random_string(prefix: str, length: int = 8) -> str:
    """Generate a random string with the given prefix."""
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}-{suffix}"


def make_project_path(project: Path | str | None, prefix: str, exist_ok: bool) -> Path:
    project_path = (
        Path(project)
        if project
        else Path(tempfile.gettempdir()) / generate_random_string(prefix)
    )
    if project_path.exists() and not exist_ok and any(project_path.iterdir()):
        raise FileExistsError(
            f"Project path {project_path} already exists and is not empty."
        )
    project_path.mkdir(parents=True, exist_ok=True)
    return project_path
