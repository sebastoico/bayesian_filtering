# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:46:24 2022

@author: CFC
"""

import numpy as np
import matplotlib.pyplot as plt

from func import rk_discrete

## Application of the Kalman Filter, the Unscented Kalman Filter and the
## Particle Filter to a Single Degree of Freedom linear system state 
## estimation

#  The system assumed in this code is a single degree of freedom linear
#  system, which is described by the equation
#
#                 d2x(t)     dx(t)               d2x_g(t)
#               m*------ + c*----- + k*x(t) = -m*--------
#                  dt2        dt                   dt2
#
#  where:
#   - m   : Mass of the system.
#   - c   : Damping coefficient of the system.
#   - k   : Stiffnes of the system.
#   - x   : Displacement of the system, its derivatives represesents the
#           velocity and the acceleration.
#   - x_g : Its second derivative represents the ground acceleration, in
#           other words, the input of the system.
#
#  The excitation signal is the El Centro earthquake.

## Response simulation

# The excitation signal
elcentro_NS = np.genfromtxt('elcentro_NS.dat')

N = len(elcentro_NS)        # number of measurements

t = elcentro_NS[:, 0]       # times
x_g = elcentro_NS[:, 1]     # ground acceleration
x_g = 9.82*x_g              # (m/s^2)

dt = t[1] - t[0]            # time between measurements

# The parameters are
m = 1                       # kN*s^2/m
c = 0.3                     # kN*s/m
k = 9                       # kN/m

# Initial state
x_0 = [0, 0]                # the system starts from rest
x_0 = np.array(x_0)
nx = len(x_0)               # number of states

x = np.zeros([N+1, nx])     # here the evolution of the system is going to
x[0, :] = x_0               # be saved

# The system is written in a state-space form
# X = [x dx/dt]'
A = [[0,       1],
     [-k/m, -c/m]]
A = np.array(A)
B = [0, 1]
B = np.array(B)

F = lambda x, u: np.dot(A, x) + np.dot(B.T, u)

# To simulate the system, the Runge-Kutta fourth order method is used
for i in range(0, N):
    x[i+1, :] = rk_discrete(F, x[i, :], x_g[i], dt)
    
# Then, the total acceleration of the system is given as
acc = -(c*x[1:, 1] + k*x[1:, 0])/m

## Measurements generation

# The filters will take the acceleration measurement as the observations
meas = np.zeros(N)
# the RMS noise-to-signal is used to add the noise
RMS = np.sqrt(np.sum(acc**2)/N)
noise_per = 0.05

meas = acc + noise_per*RMS*np.random.randn(N)

## Plots

# Acceleration measures
fig, ax = plt.subplots()

plt.ylabel('Acceleration [$m/s^{2}$]')
plt.xlabel('Time [$s$]')
plt.plot(t, acc, '-r', label='True signal')
plt.plot(t, meas, '.k', label='Measurements', markersize=5)
plt.axis([-2, 56, -2, 2])
plt.legend()

t_zoom = (t >= 32.5) & (t <= 37.5)

axins = ax.inset_axes([0.7, 0.7, 0.28, 0.28])
axins.plot(t[t_zoom], acc[t_zoom], '-r')
axins.plot(t[t_zoom], meas[t_zoom], '.k', markersize=5)
axins.axis([32.5, 37.5, -0.2, 0.2])
axins.yaxis.set_ticklabels([])
axins.xaxis.set_ticklabels([])
ax.indicate_inset_zoom(axins, edgecolor='black')

plt.show()