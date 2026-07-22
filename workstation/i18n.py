"""Language selection and Qt translation catalog loading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import QCoreApplication, QTranslator


SUPPORTED_LANGUAGES = ("zh_CN", "en_US")
LANGUAGE_ENVIRONMENT = "RURALSCAPE_LANG"
CATALOG_ROOT = Path(__file__).with_name("i18n")


def resolve_language(argv: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lang")
    options, _unknown = parser.parse_known_args(list(argv or ()))
    language = options.lang or os.environ.get(LANGUAGE_ENVIRONMENT) or "zh_CN"
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(f"Unsupported language {language!r}; choose {supported}")
    return language


def install_translation(
    application: QCoreApplication, language: str
) -> QTranslator:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    catalog = CATALOG_ROOT / f"{language}.qm"
    translator = QTranslator(application)
    if not translator.load(str(catalog)):
        raise RuntimeError(f"Unable to load translation catalog: {catalog}")
    application.installTranslator(translator)
    application._ruralscape_translator = translator
    application.setProperty("ruralscapeLanguage", language)
    return translator
