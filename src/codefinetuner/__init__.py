from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("codefinetuner")
except PackageNotFoundError:
    __version__ = "unknown"

# Expose the top-level programmatic entry point
from .pipeline import run_pipeline

# Expose subpackages explicitly so `codefinetuner.preprocess` is
# accessible after a plain `import codefinetuner` without a separate
# `import codefinetuner.preprocess`
from . import preprocess
from . import finetune 
from . import evaluate 
from . import convert

__all__ = [
    "__version__",
    "run_pipeline",
    "preprocess",
    "finetune",
    "evaluate",
    "convert",
]