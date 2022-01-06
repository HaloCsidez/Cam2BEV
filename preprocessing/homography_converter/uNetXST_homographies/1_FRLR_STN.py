# ==============================================================================
# MIT License
#
# Copyright 2020 Institute for Automotive Engineering of RWTH Aachen University.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import numpy as np

# right_front->front
# rigth_back ->right
# left_front ->left
# left_back  ->rear

# for dataaset 1_FRLR
H = [
  np.array([[ 660.228420, 0.452129, 619.723806], 
          [0.000000, 661.579398, 356.276106], 
          [0.000000, 0.000000, 1.000000]]), # front
  np.array([[ 674.732226, -1.280735, 640.149919], 
          [0.000000, 675.114779, 388.737670], 
          [0.000000, 0.000000, 1.000000]]), # rear
  np.array([[ 664.982627, 1.923222, 635.891510], 
          [0.000000, 663.965607, 356.638070], 
          [0.000000, 0.000000, 1.000000]]), # left
  np.array([[ 671.771379, 1.406491, 582.080233], 
          [0.000000, 669.243052, 373.943015], 
          [0.000000, 0.000000, 1.000000]]) # right
]
