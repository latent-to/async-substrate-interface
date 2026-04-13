import importlib.metadata
from .async_substrate import (
    AsyncQueryMapResult,
    AsyncSubstrateInterface,
    AsyncExtrinsicReceipt,
)
from .sync_substrate import QueryMapResult, SubstrateInterface, ExtrinsicReceipt


def _check_conflicts():
    """
    Verifies that py-scale-codec (`scalecodec` on PyPI) is not installed.
    """
    try:
        _ = importlib.metadata.distribution("scalecodec")
        raise RuntimeError(
            "\n\n"
            "Conflict detected: 'scalecodec' (py-scale-codec) is installed.\n"
            "This conflicts with 'cyscale', which uses the same namespace.\n\n"
            "Please remove it first:\n"
            "    pip uninstall scalecodec\n\n"
            "Then reinstall cyscale:\n"
            "    pip install cyscale\n"
        )
    except importlib.metadata.PackageNotFoundError:
        pass  # Good — scalecodec is not installed


_check_conflicts()

__all__ = [
    "AsyncQueryMapResult",
    "AsyncSubstrateInterface",
    "AsyncExtrinsicReceipt",
    "QueryMapResult",
    "SubstrateInterface",
    "ExtrinsicReceipt",
]
