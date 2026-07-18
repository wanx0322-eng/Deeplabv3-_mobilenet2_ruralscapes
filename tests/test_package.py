import importlib


def test_ruralscape_studio_package_is_importable() -> None:
    imported = importlib.import_module("ruralscape_studio")

    assert imported.__name__ == "ruralscape_studio"
