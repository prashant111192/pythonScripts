#!/usr/bin/python3

import sys
import numpy as np
import math
n = int (input('Number of probes: '))

ht = int(1)

tot = np.linspace(0,ht,n)

for i in range (n):
    tot[i] = format(tot[i], '.3f')

i=0
a=0.5
for i in range (n):
    print ('\t\t\t\t<velocity name=\"Velpt{}\">\n\t\t\t\t\t<point x=\"{}\" y=\"0\" z=\"{}\" comment=\"Measuring position\" units_comment=\"m\" />\n\t\t\t\t</velocity>'.format(i,a, tot[i]))

