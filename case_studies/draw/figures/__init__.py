"""Figure CLIs, grouped by topic into subpackages.

Each figure module keeps the file name of its historical script, which is also the name of the wrapper in
``case_studies/draw``.
"""

from __future__ import annotations

import pkgutil


def module_path(name: str) -> str:
    """Return the dotted module path of the figure module whose file is ``<name>.py``.

    Args:
        name: Module basename such as ``draw_rank_proposal_certainty``.

    Raises:
        KeyError: If no figure subpackage defines that module.
    """
    for package in pkgutil.iter_modules(__path__, f"{__name__}."):
        if not package.ispkg:
            continue
        for module in pkgutil.iter_modules([f"{__path__[0]}/{package.name.rsplit('.', 1)[1]}"]):
            if module.name == name:
                return f"{package.name}.{name}"
    raise KeyError(name)
