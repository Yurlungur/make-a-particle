#!/usr/bin/env python2

"""
make_a_wave_packet.py

Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2015-01-18 14:42:50 (jonah)>

This python script makes a particle by summing up waves of different
frequencies and saves an animation of the particle moving in a ring
cavity.

To use it, just call
python2 make_a_wave_packet.py

You can change the number of waves added up to the particle by
changing the Nmax variable.
"""

# Imports
# ----------------------------------------------------------------------
import numpy as np
from scipy import integrate
from matplotlib import animation
from JSAnimation.IPython_display import display_animation
from matplotlib import pyplot as plt
from matplotlib import rcParams
# ----------------------------------------------------------------------

# Set plotting parameters
# ----------------------------------------------------------------------
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
# ----------------------------------------------------------------------

# Global variables
# ----------------------------------------------------------------------
xmin=0 # The start of the ring cavity
xmax=2*np.pi # the end of the ring cavity
sigma=2*np.pi/25. # the width of the Guassian wave packet. (particle)
Nmax=10 # The number of waves we add up to make the wave packet.
nx = 1000 # The number of points we plot.
x=np.linspace(xmin,xmax,nx) # An array of points we plot
text_x=0.1 # x position of the text in the animation
text_y=1.  # y position of the text in the animation
text = "Waves = {}".format(Nmax) # the text in the animation
# ----------------------------------------------------------------------


# Some preliminary definitions
# ----------------------------------------------------------------------
def u(n,A,B,x,t):
    """
    A single Fourier mode, with Fourier coefficients A and B
    """
    return A*np.cos(n*(x-t)) + B*np.sin(n*(x-t))


def u0(n,x):
    """
    A Fourier sine mode.
    """
    return np.sin(n*x)


def u0_prime(n,x):
    """
    A Fourier cosine mode.
    """
    return np.cos(n*x)


def u_analytic0(x):
    """
    A Gaussian wave packet. This is the particle we want to construct.
    """
    return np.exp(-((x-np.pi)/sigma)**2)


def u_prime_analytic0(x):
    """
    This describes the motion of the particle we want to construct.
    For experts: it's the derivative of the wave packet.
    """
    return -(2.0/sigma**2)*(x-np.pi)*np.exp(-((x-np.pi)/sigma)**2)


def inner(foo,bar):
    """
    The inner (functional dot) product <foo, bar>
    """
    return integrate.quad(lambda x: (1/np.pi)*foo(x)*bar(x),xmin,xmax)[0]
# ----------------------------------------------------------------------


# The main loop, run only if you call the program from the command line
# ----------------------------------------------------------------------
if __name__=="__main__":
    # The Fourier coefficients
    Bcoeffs = [inner(lambda x: u0(i+1,x),u_analytic0)\
                   for i in range(Nmax)]
    Acoeffs = [inner(lambda x: u0_prime(i+1,x),u_analytic0)\
                   for i in range(Nmax)]

    # Some functions that contain the Fourier coefficients.
    # Defined dynamically for efficiency and readability
    def uf(x,t):
        """
        Equivalent to u above
        """
        return sum([u(i+1,Acoeffs[i],Bcoeffs[i],x,t)\
                        for i in range(len(Bcoeffs))])

    def udiff(rho):
        """
        Equivalent to uf(rho,0)
        """
        return sum(np.array([Acoeffs[i]*np.cos((i+1)*rho)
                             + Bcoeffs[i]*np.sin((i+1)*rho)\
                                 for i in range(len(Acoeffs))]))

    # Make the animation
    fig = plt.figure()
    ax = plt.axes(xlim=(0,2*np.pi),ylim=(-0.25,1.1))
    label = ax.text(text_x,text_y,text)
    line, = ax.plot([],[],lw=4)
    dt = 0.05

    def init():
        line.set_data([],[])
        return line

    def animate(i):
        t = dt*i
        y = udiff(x-t)
        line.set_data(x,y)
        return line,

    nf = int(np.ceil(2*np.pi/dt))
    anim = animation.FuncAnimation(fig,animate,
                                   init_func=init,frames=nf,
                                   interval=int(nf/10),
                                   blit=True)
    anim.save('ring_cavity_n{}.gif'.format(Nmax),writer='imagemagick')

# ----------------------------------------------------------------------

