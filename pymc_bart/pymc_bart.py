try:
    from bartrs.bartrs import PosteriorSampler, PyBartSettings, PySampler, TreeArrays
except Exception as e:
    print(f"""Warning: Could not import PyBartSettings,
           PySampler, TreeArrays, or PosteriorSampler due to: {e}""")
    PyBartSettings = PySampler = TreeArrays = PosteriorSampler = None

for cls in (PyBartSettings, PySampler, TreeArrays, PosteriorSampler):
    if cls is not None:
        cls.__module__ = "pymc_bart.pymc_bart"

__all__ = [
    name
    for name in ("PyBartSettings", "PySampler", "TreeArrays", "PosteriorSampler")
    if globals().get(name)
]
