"""Compatibility import for the modular annotation subsystem.

Set RURALSCAPE_ANNOTATE_LEGACY=1 for the one-iteration legacy fallback.
"""

from __future__ import annotations

import os

if os.environ.get("RURALSCAPE_ANNOTATE_LEGACY") == "1":
    from workstation.pages.annotate_page_legacy import (  # noqa: F401
        AnnotatePage,
        AnnotationCanvas,
        AutoLabelThread,
        ExportDialog,
        ExportThread,
    )
else:
    from workstation.pages.annotate.canvas import AnnotationCanvas  # noqa: F401
    from workstation.pages.annotate.export import (  # noqa: F401
        AutoLabelThread,
        ExportDialog,
        ExportThread,
    )
    from workstation.pages.annotate.view import AnnotatePage  # noqa: F401

__all__ = [
    "AnnotatePage",
    "AnnotationCanvas",
    "AutoLabelThread",
    "ExportDialog",
    "ExportThread",
]
