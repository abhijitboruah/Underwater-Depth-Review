"""
Where the external repositories live.

Override these if you cloned the deep-learning method repos somewhere
other than ./external. You can also set the UWDEPTH_EXTERNAL env var
to a different root.
"""

import os


_ROOT = os.environ.get('UWDEPTH_EXTERNAL', 'external')

EXTERNAL_REPOS = {
    'monodepth2':       os.path.join(_ROOT, 'monodepth2'),
    'udepth':           os.path.join(_ROOT, 'udepth'),
    'depth_anything_v2': os.path.join(_ROOT, 'depth_anything_v2'),
    # UniDepth V2 is pip-installable so it lives in site-packages,
    # not under ./external.
}
