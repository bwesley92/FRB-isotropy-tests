import warnings

def suppress_library_warnings() -> None:
    """Suppress known library warnings for cleaner output."""

    warnings.filterwarnings(
        "ignore",
        message='.*"verbose" was deprecated.*',
        category=Warning,
    )