import numpy as np
import matplotlib.pyplot as plt
import sys

from func import rk_discrete, KalmanFilter, UnscentedKalmanFilter

"""
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
"""

## ----------------------------------------------------------------------------
## Response simulation
## ----------------------------------------------------------------------------

# The excitation signal
elcentro_NS = np.genfromtxt('elcentro_NS.dat')

N = len(elcentro_NS)          # number of measurements

t = elcentro_NS[:, 0]         # times
x_g = elcentro_NS[:, 1]       # ground acceleration
x_g = 9.82*x_g                # (m/s^2)

dt = t[1] - t[0]              # time between measurements

# The parameters are
m = 1                         # kN*s^2/m
c = 0.3                       # kN*s/m
k = 9                         # kN/m

# Initial state
x_0 = [0, 0]                  # the system starts from rest
x_0 = np.array(x_0)
nx = len(x_0)                 # number of states

x = np.zeros([N+1, nx])       # here the evolution of the system is going to
x[0, :] = x_0                 # be saved

# The system is written in a state-space form
# X = [x dx/dt]'
A = [[0,       1],
    [-k/m, -c/m]]
A = np.array(A)
B = [0, 1]
B = np.array(B)

F = lambda x, u: np.dot(A, x) + np.dot(B, u)

# To simulate the system, the Runge-Kutta fourth order method is used
for i in range(0, N):
    x[i+1, :] = rk_discrete(F, x[i, :], x_g[i], dt)

# Then, the total acceleration of the system is given as
acc = -(c*x[1:, 1] + k*x[1:, 0])/m

## ----------------------------------------------------------------------------
## Measurements generation
## ----------------------------------------------------------------------------

# The filters will take the acceleration measurement as the observations
# the RMS noise-to-signal is used to add the noise
RMS = np.sqrt(np.sum(acc**2)/N)

noise_per = 0.05              # 5% of the RMS is asumed as noise variance

# The measures are generated
meas = acc #+ noise_per*RMS*np.random.randn(N)

## ----------------------------------------------------------------------------
## Kalman filter implementation
## ----------------------------------------------------------------------------

# The Kalman Filter only accepts linear sate-space equations in the form
#
#       dX/dt = A*X + B*u + q
#           y = H*X + r

# The matrices of the system are written again
# for the process
A = [[0,       1],
    [-k/m, -c/m]]
A = np.array(A)
B = [0, 1]
B = np.array(B)
# for the measurement
Hk = [-k/m, -c/m]
Hk = np.array(Hk).reshape(1, 2)

# System initial mean and covariance
x_0 = [0, 0]
x_0 = np.array(x_0)
P_0 = 0.0001*np.eye(nx)

# Process noise covariance
Q = 0.01*np.eye(nx)

# Measurement noise covariance
R = 0.01
R = np.array(R).reshape(1, 1)

# Is necessary to discretize the system, so
Ad = np.eye(A.shape[0]) + A*dt
Bd = B*dt + np.dot(A, B)*(dt**2)/2

Qd = Q*dt + (np.dot(Q, A.T) + np.dot(A, Q))*(dt**2)/2 \
    + np.dot(np.dot(A, Q), A.T)*(dt**3)/3
Rd = R/dt

# using the Kalman Filter
x_kf = np.zeros((N, 2))
x_kf[0, :] = x_0
P_kf = np.empty(N, dtype='object')
P_kf[0] = P_0

sd_kf = np.zeros_like(x_kf)

kf = KalmanFilter(A_km1 = Ad, B_km1 = Bd, H_k = Hk, x_k = x_0, \
    P_k = P_0, Q = Qd, R = Rd)

for k in range(1, N):
    kf.prediction(x_g[k])
    x_kf[k, :], P_kf[k] = kf.update(meas[k])
    sd_kf[k, :] = np.sqrt(np.diag(P_kf[k]))

## ----------------------------------------------------------------------------
## Unscented Kalman filter implementation
## ----------------------------------------------------------------------------

# The 'Unscented Kalman Filter' (UKF) is useful to the estimation
# of the nonlinear dynamical system given by:
#
#               x_{k+1} = F(x_k, u_k) + v_k        (1)
#               y_k     = H(x_k)      + n_k        (2)

# We redifine the functions
F = lambda x, u: np.dot(A, x) + np.dot(B, u)
H = lambda x: np.dot(Hk, x)

# and a "soft" discretization of the covariance matrices
Q_nl = Q*dt
R_nl = R/dt

# using the Unscented Kalman Filter

x_ukf = np.zeros((N, 2))
x_ukf[0, :] = x_0
P_ukf = np.empty(N, dtype='object')
P_ukf[0] = P_0

sd_ukf = np.zeros_like(x_ukf)

ukf = UnscentedKalmanFilter(F = F, H = H, x_0 = x_0, P_0 = P_0, \
    Rv = Q_nl, Rn = R_nl, dt = dt)
    
for k in range(0, N):
    try:
        ukf.prediction(x_g[k])
        x_ukf[k, :], P_ukf[k] = ukf.update(meas[k])
        #print(x_ukf[k, :], P_ukf[k])
        sd_ukf[k, :] = np.sqrt(np.diag(P_ukf[k]))
    except:
        print(P_ukf[k-1])
        print('Matrix not positive definite at iteration',k)
        sys.exit()
        

## ----------------------------------------------------------------------------
## Plots
## ----------------------------------------------------------------------------

# Acceleration measures
fig, ax = plt.subplots(figsize=(20, 10))

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

# Kalman filter results

plt.figure(figsize=(20, 10))
# Displacement
plt.subplot(2, 1, 1)
plt.fill_between(t, x_kf[:, 0]+sd_kf[:, 0], x_kf[:, 0]-sd_kf[:, 0], \
      color=[0.8, 0.8, 1], label='$\pm$ 1 Standard deviation')
plt.plot(t, x_kf[:, 0], '-b', label='KF')
plt.plot(t, x[1:, 0], '--r', label='True signal')
plt.legend(loc='lower right')
plt.ylabel('Displacement [$m$]')
plt.xlabel('Time [$s$]')
ymax = np.ceil(np.max(np.abs(np.concatenate((x_kf[:, 0]+sd_kf[:, 0], \
                                              x_kf[:, 0]-sd_kf[:, 0]))))*10)/10
plt.axis([np.min(t), np.max(t), -ymax, ymax])

# Velocity
plt.subplot(2, 1, 2)
plt.fill_between(t, x_kf[:, 1]+sd_kf[:, 1], x_kf[:, 1]-sd_kf[:, 1], \
      color=[0.8, 0.8, 1], label='$\pm$ 1 Standard deviation')
plt.plot(t, x_kf[:, 1], '-b', label='KF')
plt.plot(t, x[1:, 1], '--r', label='True signal')
plt.legend(loc='lower right')
plt.ylabel('Velocity [$m/s$]')
plt.xlabel('Time [$s$]')
ymax = np.ceil(np.max(np.abs(np.concatenate((x_kf[:, 1]+sd_kf[:, 1], \
                                              x_kf[:, 1]-sd_kf[:, 1]))))*10)/10
plt.axis([np.min(t), np.max(t), -ymax, ymax])

plt.show()

# Unscented Kalman filter results

plt.figure(figsize=(20, 10))
# Displacement
plt.subplot(2, 1, 1)
plt.fill_between(t, x_ukf[:, 0]+sd_ukf[:, 0], x_ukf[:, 0]-sd_ukf[:, 0], \
      color=[0.8, 0.8, 1], label='$\pm$ 1 Standard deviation')
plt.plot(t, x_ukf[:, 0], '-b', label='UKF')
plt.plot(t, x[1:, 0], '--r', label='True signal')
plt.legend(loc='lower right')
plt.ylabel('Displacement [$m$]')
plt.xlabel('Time [$s$]')
ymax = np.ceil(np.max(np.abs(np.concatenate((x_ukf[:, 0]+sd_ukf[:, 0], \
                                              x_ukf[:, 0]-sd_ukf[:, 0]))))*10)/10
plt.axis([np.min(t), np.max(t), -ymax, ymax])

# Velocity
plt.subplot(2, 1, 2)
plt.fill_between(t, x_ukf[:, 1]+sd_ukf[:, 1], x_ukf[:, 1]-sd_ukf[:, 1], \
      color=[0.8, 0.8, 1], label='$\pm$ 1 Standard deviation')
plt.plot(t, x_ukf[:, 1], '-b', label='UKF')
plt.plot(t, x[1:, 1], '--r', label='True signal')
plt.legend(loc='lower right')
plt.ylabel('Velocity [$m/s$]')
plt.xlabel('Time [$s$]')
ymax = np.ceil(np.max(np.abs(np.concatenate((x_ukf[:, 1]+sd_ukf[:, 1], \
                                              x_ukf[:, 1]-sd_ukf[:, 1]))))*10)/10
plt.axis([np.min(t), np.max(t), -ymax, ymax])

plt.show()