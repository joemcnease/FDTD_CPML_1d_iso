# /*************************************************************************/
# /* Author: Joe McNease                                                   */
# /*                                                                       */
# /* A simple introductory script to the Convolutional Perfectly           */
# /* Matched Layer (C-PML) technique.                                      */
# /*                                                                       */
# /* This example solves the 1d isotropic elastic wave equation            */
# /* using a staggered grid finite difference scheme (Madriaga 1976,       */
# /* Vireaux 1984/1986, Levander 1988, etc.). The staggered grid           */
# /* is a very elegant scheme and used extensively in numerical            */
# /* modeling of seismic waves. There are many other ways of modeling      */
# /* these waves, such as psuedospectral, finite element, etc., but        */
# /* finite differences are probably the easiest to understand and         */
# /* implement.                                                            */
# /*                                                                       */
# /* The C-PML is an extension of the PML method by Berrenger (1995)       */
# /* which was originally created for modeling electromagnetic waves.      */
# /* However, this technique called for split fields (splitting the        */
# /* divergence operator into normal and parallel components) and suffered */
# /* when waves were at grazing incidence to the PML boundary. Roden and   */
# /* Gedney (2000) increased the effectiveness of the PML with their       */
# /* C-PML, using a recursive relation and memory variables to quickly     */
# /* and efficiently calculate the convolutional term (resulting from      */
# /* inverse Fourier transform with frequency dependence) in the PML       */
# /* layers. Komatitsch and Martin (2007) then applied this using unsplit  */
# /* fields, which is implemented here. This is simple to add to existing  */
# /* finite difference codes, which you can see in the code below!         */
# /*                                                                       */
# /* Try changing the material parameters or extending the code to 2d      */
# /* as a challenge!                                                       */
# /*************************************************************************/


import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.animation import FuncAnimation


# Time and space discretization
nt = 2500                    # Number of time steps
dt = 0.0001                  # delta t [s]
nx = 1000                    # Number of grid points
dx = 1                       # delta x [m]
t = np.arange(0., nt*dt, dt) # s
x = np.arange(0., nx*dx, dx) # m

# Material parameters
rho = 2800                   # kg/m^3
c = 4500                     # m/s
mu = c**2 * rho              # Pa

# C.F.L stability criterion
cfl = c*dt/dx
if (cfl > 1):
    sys.exit("Scheme is not stable. Increase grid size or decrease time step.")

# Initial conditions for fields
vytot  = np.zeros((nt, x.shape[0]))
vy     = np.zeros_like(x)
vyOld  = np.zeros_like(x)
dydx   = np.zeros_like(x)
tyx    = np.zeros_like(x)
tyxOld = np.zeros_like(x)
dtyxdx = np.zeros_like(x)

# Source time function parameters
sx = nx//2
f0 = 200
t0 = 5/f0
stf = np.exp(-f0**2 * (t - t0)**2) # Gaussian stf
stf = -np.diff(np.diff(stf))
stf = np.append(stf, (0, 0))      # 2nd derivative (ricker)

# Example Gibb's phenomena for jump discontinuity (Square wave)
#stf = np.where((t<t0) | (t>2*t0), 0, 1)

# CPML boundary condition parameters
damping      = np.zeros_like(x)
alpha        = np.zeros_like(x)
dydxMemory   = np.zeros_like(x)
dtyxdxMemory = np.zeros_like(x)
bx           = np.zeros_like(x)
ax           = np.zeros_like(x)

# Receivers
rlx = np.array([300, 700])          # Locations in meters
seis = np.zeros([rlx.shape[0], nt]) # Seismograms at receiver locations

# Get grid locations for receivers
rlidx = np.zeros_like(rlx)
for i in range(len(rlx)):
    rlidx[i] = (np.argmin(np.abs(x-rlx[i])))

Rc = 0.001                      # Reflection coefficient
kappa = 1                       # Scaling of differential operator
N = 2                           # Exponential scaling of profile
L = 200                         # CPML layer thickness
cp = c                          # Wave velocity [m/s]
d0 = -(N+1)*cp*np.log(Rc)/(2*L) # ~341.9
alpha_max = np.pi*f0 / 10

# Create damping and memory profiles for CMPL
for i in range(nx):
    if (i*dx > (dx*nx - L*dx)):
        damping[i] = d0*((i-(nx-L))/L)**N
        alpha[i]   = ((L+nx-i)/L - 1)*alpha_max
        bx[i]      = np.exp(-(damping[i]/kappa + alpha[i])*dt)
        ax[i]      = damping[i]*(bx[i]-1)/(kappa*(damping[i] + kappa*alpha[i]))

# Print some info about the modeling
print("-"*50)
print(f"CFL Number            : {cfl}")
print(f"SH Wave Velocity      : {c}")
print(f"Rho                   : {rho}")
print(f"Mu                    : {mu}")
print(f"Wavelength            : {c/f0}")
print(f"Points per wavelength : {c/f0/dx}")
print("-"*50)

# Plot source time function. Make sure there are no
# jump discontinuities or you will get ringing!
fig = plt.figure()
plt.plot(t, stf)
plt.show()

for it in range(nt-1):
    # Compute stress derivatives and update velocity
    for i in range(1, nx-1):
        dydxMemory[i] = bx[i]*dydxMemory[i] + ax[i]*dydx[i]
        dtyxdx[i] = ((1/kappa)*(tyx[i+1] - tyx[i])/dx) + dtyxdxMemory[i]
    vy = vyOld + (dt/rho)*dtyxdx
    vy[sx] += stf[it]*dt/rho # Add source
    seis[:, it] = vy[rlidx]  # Get seismograms at receivers
    vyOld = vy

    # boundary conditions
    vy[:2] = 0.
    vy[-2:] = 0.

    # Save velocity field for animation
    vytot[it, :] = vy

    # Compute velocity derivatives and update stress
    for i in range(1, nx-1):
        dtyxdxMemory[i] = bx[i]*dtyxdxMemory[i] + ax[i]*dtyxdx[i]
        dydx[i] = ((1/kappa)*(vy[i] - vy[i-1])/dx) + dydxMemory[i]
    tyx = tyxOld + (dt*mu)*dydx
    tyxOld = tyx

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 9))
line, = axs[0].plot(x, vy)
vm = max(np.abs(np.min(vytot)), np.abs(np.max(vytot)))
vmin, vmax = -vm-vm*0.1, vm+vm*0.1
axs[0].set_ylim([vmin, vmax])
axs[0].set_xlim([0, nx*dx])
axs[0].vlines(nx*dx-L*dx, vmin, vmax, colors='r', linestyles='--')
scat = axs[0].scatter(rlx, np.zeros_like(rlx), marker='^',
                      s=60, facecolor='g')

seis1, = axs[1].plot(t, seis[0])
axs[1].set_ylim([vmin, vmax])
axs[1].set_xlim([0, nt*dt])
at = AnchoredText("Receiver [x=300]", prop=dict(size=15),
                  frameon=True, loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
axs[1].add_artist(at)

seis2, = axs[2].plot(t, seis[1])
axs[2].set_ylim([vmin, vmax])
axs[2].set_xlim([0, nt*dt])
at = AnchoredText("Receiver [x=700]", prop=dict(size=15),
                  frameon=True, loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
axs[2].add_artist(at)

def animate(i):
    axs[0].set_title(f"Time step: {i*dt:.3f}")
    line.set_ydata(vytot[i, :])
    seis1.set_data(t[:i], seis[0][:i])
    seis2.set_data(t[:i], seis[1][:i])
    scat.set_offsets(list(zip(rlx, vytot[i, :][rlidx])))

    return line,


ani = FuncAnimation(fig, animate, repeat=True, frames=t.shape[0]-1,
                    interval=0.1)

plt.show()
