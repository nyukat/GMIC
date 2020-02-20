# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Defines constants used in src.
"""

class VIEWS:
    L_CC = "L-CC"
    R_CC = "R-CC"
    L_MLO = "L-MLO"
    R_MLO = "R-MLO"

    LIST = [L_CC, R_CC, L_MLO, R_MLO]

    @classmethod
    def is_cc(cls, view):
        return view in (cls.L_CC, cls.R_CC)

    @classmethod
    def is_mlo(cls, view):
        return view in (cls.L_MLO, cls.R_MLO)

    @classmethod
    def is_left(cls, view):
        return view in (cls.L_CC, cls.L_MLO)

    @classmethod
    def is_right(cls, view):
        return view in (cls.R_CC, cls.R_MLO)


INPUT_SIZE_DICT = {
    VIEWS.L_CC: (2944, 1920),
    VIEWS.R_CC: (2944, 1920),
    VIEWS.L_MLO: (2944, 1920),
    VIEWS.R_MLO: (2944, 1920),
}

PERCENT_T_DICT = {
    "1":0.02,
    "2":0.03,
    "3":0.03,
    "4":0.05,
    "5":0.1}