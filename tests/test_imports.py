import importlib

MODULES = [
    "src.data.manifest",
    "src.preprocessing.preprocess",
    "src.features.extract_features",
    "src.models.detector",
    "src.models.train",
    "src.eval.metrics",
    "src.eval.harness",
    "src.explain.rationale_agent",
]


def test_all_modules_import():
    for name in MODULES:
        importlib.import_module(name)
