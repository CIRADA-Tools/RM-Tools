from io import StringIO
import numpy as np
c = StringIO("1,0,2\n3,0,4")
x = np.loadtxt(c, delimiter=',', usecols=(0, 2))
print(x)