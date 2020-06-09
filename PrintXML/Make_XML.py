#!/usr/bin/python3

import sys
import numpy as np
import math

n = int(input('Number of elements: '))
dia = int(input('diameter of the digester: ')) 
ht = input('height to be probed: ')
if ht.find('.')!=-1:
    float(ht)
    format(ht, '3')
else:
    int(ht)
if (n%2) == 1:
    n=n-1

d_l = dia*(-0.5)
d_u = dia*(0.5)
#a = -0.175 
#b = 0.175

# l_b = np.linspace(d_l, a, n/2)
# u_b = np.linspace(b, d_u, n/2)
tot = np.linspace(d_l, d_u, n)

i=0
for i in range (n): 
    tot[i] = format(tot[i], '.3f')
i=0

for i in range(n):
    print ('\t\t\t\t<velocity name=\"17Velpt{}\">\n\t\t\t\t\t<point x=\"{}\" y=\"0\" z=\"{}\" comment=\"Measuring position\" units_comment=\"m\" />\n\t\t\t\t</velocity>'.format(i,tot[i],ht))
