tokens = 10

import numpy as np

a = [i+1 for i in range(tokens)]

b = {i:[] for i in range(tokens)}

b[2].append(1)

b[2].append(2)

print(b)

del b[2][0]

b[2].append(3)

del a[3]

print(a)

print(b)