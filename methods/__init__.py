"""
Method registry. Each entry maps a short key to a callable predictor
(class instance with a .predict(rgb) -> depth method).

Most methods need configuration (weight paths, dataset-specific beta_eff,
device) which is supplied by run.py. The factory function builds the
predictor on demand.
"""

from . import udcp, peng_cosman


_REGISTRY = {
    'udcp':     ('methods.udcp',                'UDCP'),
    'pc':       ('methods.peng_cosman',         'PengCosman'),
    'md2':      ('methods.monodepth2',          'Monodepth2'),
    'udepth':   ('methods.udepth',              'UDepth'),
    'dpt':      ('methods.dpt',                 'DPT'),
    'midas':    ('methods.midas',               'MiDaS'),
    'dav2':     ('methods.depth_anything_v2',   'DepthAnythingV2'),
    'unidepth': ('methods.unidepth_v2',         'UniDepthV2'),
}


def build(method_key, **kwargs):
    """Construct the predictor for `method_key`.

    Deep methods are imported lazily so we don't have to install all
    deep-learning deps to run UDCP/PC.
    """
    if method_key not in _REGISTRY:
        raise ValueError(f'unknown method: {method_key}. '
                         f'Available: {sorted(_REGISTRY)}')
    module_name, class_name = _REGISTRY[method_key]
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def available():
    return sorted(_REGISTRY)
