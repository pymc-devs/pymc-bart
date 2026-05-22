try:
    from pymc_bartrs.pymc_bartrs import PyBartSettings, PySampler, TreeArrays
except Exception:
    PyBartSettings = PySampler = TreeArrays = None
    
for cls in (PyBartSettings, PySampler, TreeArrays):
    if cls is not None:
        cls.__module__ = "pymc_bart.pymc_bart"

__all__ = [name for name in ("PyBartSettings", "PySampler", "TreeArrays") if globals().get(name)]