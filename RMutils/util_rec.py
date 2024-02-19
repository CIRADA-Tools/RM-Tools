#!/usr/bin/env python
# =============================================================================#
#                                                                             #
# NAME:     util_rec.py                                                       #
#                                                                             #
# PURPOSE:  Functions for operating on python record arrays.                  #
#                                                                             #
# MODIFIED: 19-November-2015 by C. Purcell                                    #
#                                                                             #
# CONTENTS:                                                                   #
#                                                                             #
#  pyify                    ... return type converters given type strings     #
#  irecarray_to_py          ... convert a recarray into a list                #
#  fields-view              ... return a view of chosen fields in a recarray  #
#                                                                             #
# =============================================================================#
#                                                                             #
# The MIT License (MIT)                                                       #
#                                                                             #
# Copyright (c) 2015 Cormac R. Purcell                                        #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the "Software"),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
# =============================================================================#


import numpy as np


# -----------------------------------------------------------------------------#
def pyify(typestr):
    """
    Return a Python :class:'type' that most closely represents the
    type encoded by *typestr*
    """
    if typestr[1] in "iu":
        return int
    elif typestr[1] == "f":
        return float
    elif typestr[1] == "S":
        return str
    return lambda x: x


# -----------------------------------------------------------------------------#
def irecarray_to_py(a):
    """
    Slow conversion of a recarray into a list of records with python types.
    Get the field names from :attr:'a.dtype.names'.
    :Returns: iterator so that one can handle big input arrays
    """
    pytypes = [pyify(typestr) for name, typestr in a.dtype.descr]

    def convert_record(r):
        return tuple([converter(value) for converter, value in zip(pytypes, r)])

    return (convert_record(r) for r in a)


# -----------------------------------------------------------------------------#
def fields_view(arr, fieldNameLst=None):
    """
    Return a view of a numpy record array containing only the fields names in
    the fields argument. 'fields' should be a list of column names.
    """

    # Default to all fields
    if not fieldNameLst:
        fieldNameLst = arr.dtype.names
    dtype2 = np.dtype({name: arr.dtype.fields[name] for name in fieldNameLst})

    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)
