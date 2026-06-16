#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fault module for fault geometry modeling and discretization.

Provides classes for:
- Planar faults (Fault)
- Multiple fault segments (MultiFault)
- Rectangular patch discretization (RectPatch)
- Triangular patch discretization (TriPatch)

@author: Zelong Guo
"""

__author__ = "Zelong Guo"

from .fault import Fault
from .multifault import MultiFault
from .rectpatch import RectPatch
from .tripatch import TriPatch

__all__ = ['Fault', 'MultiFault', 'RectPatch', 'TriPatch']
