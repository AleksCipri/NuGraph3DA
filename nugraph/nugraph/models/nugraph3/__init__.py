"""NuGraph3 model module"""
from .nugraph3 import NuGraph3



"""NuGraph3_da model module
    Old one with DA just in the event decoder and no labels from target
"""
from .nugraph3_da import NuGraph3DA

"""NuGraph3_da model module
    New one with DA in the event decoder and optionally in semantic. Labels are used from both datasets.
"""
from .nugraph3_da_event_semantic import NuGraph3DA  

