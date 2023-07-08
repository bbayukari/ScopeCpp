#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Site    :
# @File    : __init__.py

__version__ = "0.0.1"
__author__ = ("Zezhi Wang, Jin Zhu,"
              "Kangkang Jiang, Junhao Huang,"
              "Junxian Zhu, Xueqin Wang")
              

from .solver import (ScopeSolver)
from .other_solver import (BaseSolver, HTPSolver, GraspSolver)

__all__ = [
    "ScopeSolver",
    "HTPSolver",
    "GraspSolver",
]
