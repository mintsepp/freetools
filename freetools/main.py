import numpy as np
import freetools as ft

'''
Main program to do quick plots etc.

Here I have an example program to plot
Fried parameter over 0 to 60 degree zenith angle.
'''

k = 2*np.pi/1550e-9         # Wavenumber for 1550nm
h_sat = 0.5e6               # LEO satellite altitude
zeta = np.linspace(0,60)    # Zenith angle from 0 to 60
r0 = ft.fried_param(k,h_sat,zeta)

ft.quickPlot(zeta,r0,"Fried parameter for 1550nm","1550nm","zenith angle [deg]","r0 [m]")