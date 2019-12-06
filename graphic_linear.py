#!/usr/bin/python
from __future__ import print_function
from dolfin import *
import numpy as np
import math
import getopt, sys
import matplotlib.pyplot as plt
from mpltools import annotation

p1Eu=np.loadtxt("energy_linear_p1.txt")
p1Lu=np.loadtxt("L2u_linear_p1.txt")
p1Lw=np.loadtxt("L2w_linear_p1.txt")


p2Eu=np.loadtxt("energy_linear_p2.txt")
p2Lu=np.loadtxt("L2u_linear_p2.txt")
p2Lw=np.loadtxt("L2w_linear_p2.txt")

N,M=np.shape(p1Eu)

xh=p1Eu[0,1:M]

#print()
plt.figure(1)
line_D1=plt.plot(np.log(xh),np.log(p1Eu.diagonal()[1::]),label=r'(P1)Energy',ls='-.')
line_D2=plt.plot(np.log(xh),np.log(p1Lu.diagonal()[1::]),label=r'(P1)$L_2$ of $u$',ls='--')
line_D3=plt.plot(np.log(xh),np.log(p1Lw.diagonal()[1::]),label=r'(P1)$L_2$ of $w$')


annotation.slope_marker((5.0,-7),-1,invert=True)
annotation.slope_marker((5.0,-13.5),-2,invert=True)

line_V1=plt.plot(np.log(xh),np.log(p2Eu.diagonal()[1::]),label=r'(P2)Energy',marker="^")
line_V2=plt.plot(np.log(xh),np.log(p2Lu.diagonal()[1::]),label=r'(P2)$L_2$ of $u$',marker=".")
line_V3=plt.plot(np.log(xh),np.log(p2Lw.diagonal()[1::]),label=r'(P2)$L_2$ of $w$',marker="s")
plt.xlabel(r'$\log(1/h)$')
plt.legend(loc='best')

plt.ylabel('log(error)')
plt.title('Numerical results of linear basis')
#plt.show()


for j in range(1,M): print('%5.4e' % p1Lu[N-1,j])