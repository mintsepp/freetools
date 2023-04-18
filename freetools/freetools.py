import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import scipy.integrate as integrate
import scipy.special as special
from aotools.functions.zernike import zernIndex

'''
Toolbox for NWO Optical Superhighways WP1.

Andrew's book refers to "Laser Beam Propagation through Random Media"
2nd edition by Larry C. Andrews and Ronald L. Phillips.

The purpose of this toolbox is to be able to compare simulation results
to theoretical results, without having to re-program the equations.

I have also added some plotters to this toolbox to help visualize FAST data.

Any questions about the toolbox can be sent to seppala(at)strw.leidenuniv.nl.
'''

#TODO: Add natural constants such as radius of the earth and speed of light for easy use
#TODO: Add comments inside each function for better documentation
#TODO: Split functions thematically into their own files

#Integral in mu_u1 requires high precision, hence using mpmath.
#Using scipy's integral or lower precision results in a jagged integral.
mp.mp.dps = 16

'''Geometrical path length:
Calculates the path length to satellite from the surface of the Earth
to a satellite at altitude h_sat with given zenith angle zeta in degrees.'''
def l_path(h_sat, zeta):
    r_earth = 6.371009e6
    a = 1
    b = -2 * r_earth * np.cos(np.pi - zeta* np.pi / 180)
    c = r_earth ** 2 - (r_earth + h_sat) ** 2
    r1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    r2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    if r1 >= 0:
        return r1
    else:
        return r2

'''Short notation for sec^2(x):'''
def sec2(x):
    return 1.0 / np.cos(x)**2

'''Cn2 values from HV57 model:
Eq. 12.1 from Andrews' book. Gives Cn2 value as a function of altitude h
based on the Hufnagel-Valley-57 model. w is the pseudowind and A is a
nominal value of Cn2(0) at the ground.'''
def HV57(h, w=21, A=1.7e-14):
    return 0.00594 * (w / 27) ** 2 * (1e-5 * h) ** 10 * mp.exp(-h / 1000) + 2.7e-16 * mp.exp(
            -h / 1500) + A * mp.exp(-h / 100.)

'''Integrated Cn2 from HV57 model:
Calculates the integrated Cn2 value based on the HV57 model over 0 to given
altitude h.'''
def integrated_cn2(h, w=21, A=1.7e-14):
    h0 = 0
    func = lambda h: 0.00594 * (w / 27) ** 2 * (1e-5 * h) ** 10 * mp.exp(-h / 1000) + 2.7e-16 * mp.exp(
        -h / 1500) + A * mp.exp(-h / 100.)
    integral = mp.quad(func, [h0, h])
    return float(integral.real)

'''Fried parameter r0:
Eq. 14.25 from Andrews' book. Calculates Fried parameter r0 with the HV57 model
for a wavenumber k.'''
def fried_param(k, h, zeta=0):
    cn2 = integrated_cn2(h)
    return (0.42*k**2*1/np.cos(np.deg2rad(zeta))*cn2)**(-3/5)

'''Zenith correction:
Calculates zenith correction from 1/sec(x). Zenith angle is in degrees.'''
def calc_zenith_correction(zenith_angle):
    zenith_angle_rads = np.radians(zenith_angle)
    gamma = 1/np.cos(zenith_angle_rads)
    return gamma

'''Greenwood frequency:
Denotes the frequency required for optimal correction in AO system. Can be thought
as the frequency of the changes in the turbulent atmosphere. Here v_w is the constant
wind speed.'''
def greenwood_freq(k, zeta, h, h0=0):
    lam = 2*np.pi/k
    zeta = np.deg2rad(zeta)
    func = lambda x : HV57(x)*bufton_wind(zeta,x)**5/3
    integral = mp.quad(func,[h0,h])
    result = 2.31*lam**(-6/5)*(1/mp.cos(zeta)*integral)**(3/5)
    return result.real

'''Bufton wind model:
Gives the wind speed according to Bufton wind model. Can be used as the generalized
model if set vg (wind speed at ground), vt (wind speed at tropopause), ht (altitude
of the tropopause) and Lt (thickness of the tropopause layer) by user.'''
def bufton_wind(zeta, h, vg=8, vt=30, ht=9400., Lt=4800.):
    return vg + vt * mp.exp(-((h*mp.cos(zeta) - ht) / Lt) ** 2)

'''Scintillation index for plane wave without aperture averaging:
Eq. 12.38 from Andrews' book. Scintillation index for an unbounded plane wave
(theta = 1, lambda = 0). Uses HV57 model. Zenith angle zeta is given in degrees.'''
def scint_index38(k, zeta, h0, h):
    func = lambda x : HV57(x)*(x-h0)**(5/6)
    integral = mp.quad(func, [h0,h])
    result = 2.25*k**(7/6)*1/(np.cos(zeta*np.pi/180)**(11/6))*integral
    return result.real

'''Phase variance for slant path:
Eq. 8.124 (double integral) from Andrews' book. Calculates phase variance
for a slant path using the HV57 model and Kolmogorov power spectrum. k here
is kappa (spatial frequency) and wavk is wave number, set to 1550nm by default.'''
def phs_var(h, wavk=4.0537e6, L0=25, l0=1e-9, zeta=0):
    cn2 = integrated_cn2(h)*calc_zenith_correction(zeta)
    C = 2 * np.pi
    km = 5.92 / l0
    k0 = C / L0
    kmax = mp.inf
    func = lambda k : k*0.033*cn2*np.exp(-k**2/km**2) / ((k**2 + k0**2)**(11.0/6.0))
    integral = integrate.quad(func, 0, kmax)
    return 4*np.pi**2 * wavk**2 * integral[0]

'''Piston removed phase variance:
Eq. 14.91 from Andrews' book. Phase variance with piston removed
(Zernike mode 1, m = n =0).'''
def piston_removed_phs_var(h, wavk=4.0537e6, L0=30, l0=0.01, zeta=0):
    cn2 = integrated_cn2(h) * calc_zenith_correction(zeta)
    dg = 0.5 #ground aperture
    C = 2 * np.pi
    km = 5.92 / l0
    k0 = C / L0
    kmax = mp.inf
    func = lambda k: k * 0.033 * cn2 * np.exp(-k ** 2 / km ** 2) / ((k ** 2 + k0 ** 2) ** (11.0 / 6.0))\
                     * (1-np.exp(-k**2*dg**2/16))
    integral = integrate.quad(func,0,kmax)
    return 4 * np.pi ** 2 * wavk ** 2 * integral[0]

'''Tilt removed phase variance:
Eq. 14.93 from Andrews' book. Phase variance with tip-tilt removed
(Zernike modes 2 (m = -1, n = 1) and 3 (m = n = 1).'''
def tilt_removed_phs_var(h, wavk=4.0537e6, L0=30, l0=0.01, zeta=0):
    cn2 = integrated_cn2(h) * calc_zenith_correction(zeta)
    dg = 0.5 #ground aperture
    C = 2 * np.pi
    km = 5.92 / l0
    k0 = C / L0
    kmax = mp.inf
    func = lambda k: k * 0.033 * cn2 * np.exp(-k ** 2 / km ** 2) / ((k ** 2 + k0 ** 2) ** (11.0 / 6.0))\
                     * (special.jv(2,k*dg/2)/k*dg/2)**2
    integral = integrate.quad(func,0,kmax)
    return 64 * np.pi ** 2 * wavk ** 2 * integral[0]

'''Piston and tilt removed phase variance:
Eq. 14.94 from Andrews' book. Removing tip-tilt from piston removed gives both removed.
Essentially phase variance after correcting for first 3 Zernike modes.'''
def piston_tilt_phs_var(h, wavk=4.0537e6, L0=30, l0=0.01, zeta=0):
    return piston_removed_phs_var(h, wavk, L0, l0, zeta) - tilt_removed_phs_var(h, wavk, L0, l0, zeta)

#TODO: this does not quite work yet. Agrees with simulation up to ~50 modes.
'''Phase variance after AO correction:
Calculates phase variance after doing n_noll number of corrections to the phase variance
based on HV57 turbulence model and Kolmogorov power spectrum. Modes_phs is first the
phase variance caused by turbulence, and then remove variance using the filter functions
which are coded in the zernike_squared_filter(...).'''
def zernike_modes_phase(h, wavk=4.0537e6, L0=30, l0=0.01, zeta=0, n_noll=0, dg=0.5):
    modes_phs = phs_var(h, wavk, L0, l0, zeta)
    if n_noll == 0:
        return modes_phs
    for i in range(1,n_noll+1):
        modes_phs = modes_phs - zernike_squared_filter(h, wavk, L0, l0, zeta, i, dg)
    return modes_phs

#TODO: this does not quite work yet. Agrees with simulation up to ~50 modes.
'''Zernike squared filter:
Eq. 14.86 from Andrews' book. Filter functions for the phase variance. Uses the
zernIndex(n_noll) from AOtools to keep same numbering for m,n as FAST has. Jv
here is the Bessel functions of the first kind from Scipy library, which are also
used by FAST. The phase variance eq. 8.124 here is in cartesian form.'''
def zernike_squared_filter(h, wavk=4.0537e6, L0=25, l0=0.01, zeta=0, n_noll=0, dg=0.5):
    n, m = zernIndex(n_noll)
    cn2 = integrated_cn2(h) * calc_zenith_correction(zeta)
    C = 2 * np.pi
    km = 5.92 / l0
    k0 = C / L0
    kmax = np.inf

    func = lambda kx, ky: 0.033 * cn2 * np.exp(-(kx**2+ky**2) / km ** 2) / (((kx**2+ky**2) + k0 ** 2) ** (11.0 / 6.0)) \
                    * (n+1)*(2 * special.jv((n + 1), np.sqrt(kx**2+ky**2) * dg / 2) / (np.sqrt(kx**2+ky**2) * dg / 2)) ** 2 * 2 * np.sin(m * np.arctan2(kx,ky)) ** 2
    if m == 0:
        func = lambda kx, ky: 0.033 * cn2 * np.exp(-(kx**2+ky**2) / km ** 2) / (((kx**2+ky**2) + k0 ** 2) ** (11.0 / 6.0)) \
                    * (n + 1) * (2 * special.jv((n + 1), np.sqrt(kx**2+ky**2) * dg / 2) / (np.sqrt(kx**2+ky**2) * dg / 2)) ** 2
    elif n_noll % 2 == 0:
        func = lambda kx, ky: 0.033 * cn2 * np.exp(-(kx**2+ky**2) / km ** 2) / (((kx**2+ky**2) + k0 ** 2) ** (11.0 / 6.0)) \
                    * (n+1)*(2 * special.jv((n + 1), np.sqrt(kx**2+ky**2) * dg / 2) / (np.sqrt(kx**2+ky**2) * dg / 2)) ** 2 * 2 * np.cos(m * np.arctan2(kx,ky)) ** 2

    integral = integrate.dblquad(func,-kmax,0,-kmax,0)+integrate.dblquad(func,0,kmax,0,kmax)

    return 2 * np.pi * wavk ** 2 * integral[0]

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

'''
Following equations are based on paper "Strehl ratio and scintillation theory
for uplink Gaussian-beam waves: beam wander effects" by L. Andrews, et al. 2006'''

'''RMS angular beam wander:
Eq. 19. Calculates the root-mean square angular beam wander in radians as a function
of launch beam radius.'''
def r_c2(h_0, h, zeta, w_0):
    zeta = np.radians(zeta)
    result = 0.54 * (h - h_0) ** 2 * sec2(zeta)*(1.692*np.pi**2*1/np.cos(zeta))/(2*w_0)**(1/3)*integrated_cn2(l_path(h,zeta))
    return result

'''Uncorrected Strehl ratio:
Eq. 26. Calculates the uncorrected Strehl ratio. Strehl ratio measures the quality of
an optical image. Values are between 0 and 1, with 1 being perfcectly unaberrated image.
ratio here is W0/r0, so the ratio between beam launch aperture and Fried parameter.'''
def strehl(ratio):
    result = (1+2**(5/2)*ratio**(5/3))**(-6/5)
    return result

'''Tip-tilt corrected Strehl ratio:
Eq. 27. Calculates Strehl ratio after tip-tilt correction.'''
def strehl_tilt(ratio):
    result = (1+(5.56-4.84/(1+0.04*ratio**(5/3)))*ratio**(5/3))**(-6/5)
    return result

'''Beam radius at the receiver plane:
Calculates the beam radius at the receiver plane, given the launch beam radius w_0.
Calculated for a collimated beam, so theta_0 = 1.'''
def w_r(h, zeta, lamda, w_0):
    theta_0 = 1
    result = w_0*np.sqrt(theta_0**2+lampda_0(h,zeta,lamda,w_0)**2)
    return result

'''On-axis scintillation index:
Eq. 32. Calculates the on-axis scintillation index. Angle is in radians.'''
def si_on_axis(h_0, h, zeta, lamda, w_0):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    result = si_on_axis_rytov(h_0, h, zeta, lamda, w_0) + 5.95 \
             * (h - h_0) ** 2 * sec2(zeta) * (2 * w_0 / r_0) ** (5 / 3) \
             * (alpha_pe(h_0, h, zeta, lamda, w_0) / w_r(h, zeta, lamda, w_0)) ** 2
    return result

'''On-axis scintillation index based on Rytov theory:
Eq. 37. Calculates the on-axis scintillation index that is based on the conventional
Rytov theory, which predicts that the on-axis scintillation continually decreases
as the beam launch aperture increases. Angle is in radians.'''
def si_on_axis_rytov(h_0, h, zeta, lamda, w_0): # eq 37
    k = 2 * np.pi / lamda
    result = 8.70*mu_u1(h_0, h,zeta,lamda,w_0)*k**(7/6)*(h-h_0)**(5/6)*(1/np.cos(zeta))**(11/6)
    return result

'''On-axis scintillation index for strong fluctuations:
Eq. 44. Under strong irradiance fluctuations where launch beam aperture is larger than
the Fried parameter (w0 > r0), weak fluctuations (eq. 32) are not adequate.'''
def si_on_axis_strong(h_0, h, zeta, lamda, w_0): # eq 44
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    s_b = np.sqrt(si_on_axis_rytov(h_0, h, zeta, lamda, w_0))
    result = 5.95 * (h - h_0) ** 2 * sec2(zeta) * (2 * w_0 / r_0) ** (5 / 3) * (alpha_pe(h_0, h, zeta, lamda, w_0) / w_r(h, zeta, lamda, w_0)) ** 2\
    + mp.exp(0.49*s_b**2/(1+0.56*(1+theta_r(h, zeta, lamda, w_0))*s_b**(12/5))**(7/6)\
             +0.51*s_b**2/(1+0.69*s_b**(12/5)))-1
    return result

'''Off-axis scintillation index:
Eq. 45. Calculates off-axis scintillation index as a function of path length and
radial distance from the center of the beam (r), based on Andrews' theory.'''
def si_strong(h_0, h, zeta, lamda, w_0, r):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    alpha_ratio = r/l_path(h,zeta) - alpha_pe(h_0, h, zeta, lamda, w_0)
    u = 0
    if alpha_ratio > 0:
        u = 1
    result = 5.95*(h-h_0)**2*sec2(zeta)*(2*w_0/r_0)**(5/3)*(alpha_ratio/w_r(h, zeta, lamda, w_0))**2*u\
    + si_on_axis_strong(h_0, h, zeta, lamda, w_0)
    return result

'''Off-axis scintillation index based on Rytov theory:
Eq. 38. Calculates off-axis scintillation index as a function of path length and
radial distance from the center of the beam (r), based on Rytov theory.'''
def si_rytov(h_0, h, zeta, lamda, w_0, r):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    alpha_ratio = r / l_path(h, zeta) - r_c2(h_0, h, zeta, w_0)**(1/2) / l_path(h, zeta)**2
    u = 0
    if alpha_ratio > 0:
        u = 1
    result = 8.70*mu_u1(h_0, h,zeta,lamda,w_0)*k**(7/6)*(h-h_0)**(5/6)*(1/np.cos(zeta))**(11/6)\
    + 5.95 * (h-h_0) ** 2 * sec2(zeta) *(2*w_0/r_0) ** (5/3) * (alpha_ratio/w_r(h, zeta, lamda, w_0)) ** 2 * u
    return result

'''On-axis scintillation index of a tracked beam:
Eq. 42. Calculates the on-axis scintillation index of a tracked, collimated beam.'''
def si_on_axis_tracked(h_0, h, zeta, lamda, w_0):
    sigma_b = np.sqrt(si_on_axis_rytov(h_0, h, zeta, lamda, w_0))
    result = mp.exp(0.49*sigma_b**2 / (1+0.56*(1+theta_r(h, zeta, lamda, w_0))*sigma_b**(12/5))**(7/6)\
                    + 0.51*sigma_b**2/(1+0.69*sigma_b**(12/5))**(5/6))-1
    return result

'''Off-axis scintillation index of a tracked beam:
Eq. 43. Calculates the off-axis scintillation index of a tracked, collimated beam as
a function of path length and radial distance from the center of the beam.'''
def si_tracked(h_0, h, zeta, lamda, w_0, r):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    alpha_ratio = r/l_path(h,zeta) - r_c2(h_0, h, zeta, w_0)**(1/2)/l_path(h,zeta)
    u = 0
    if alpha_ratio > 0:
        u = 1
    result = 5.95 * (h-h_0) ** 2 * sec2(zeta) * (2*w_0/r_0) ** (5/3) * (alpha_ratio/w_r(h, zeta, lamda, w_0)) ** 2 * u \
             + si_on_axis_rytov(h_0, h, zeta, lamda, w_0)
    return result

'''On-axis scintillation index with strong fluctuations for a tracked beam:
Eq. 44. Calculates on-axis scintillation index for a tracked beam when the beam
launch aperture is larger than the Fried parameter (strong fluctuations).'''
def si_on_axis_strong_tracked(h_0, h, zeta, lamda, w_0):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    s_b = np.sqrt(si_on_axis_rytov(h_0, h, zeta, lamda, w_0))
    result = 5.95 * (h - h_0) ** 2 * sec2(zeta) * (2 * w_0 / r_0) ** (5 / 3)\
             * (alpha_petc(h, zeta, lamda, w_0) / w_r(h, zeta, lamda, w_0)) ** 2\
    + mp.exp(0.49*s_b**2/(1+0.56*(1+theta_r(h, zeta, lamda, w_0))*s_b**(12/5))**(7/6)\
             +0.51*s_b**2/(1+0.69*s_b**(12/5)))-1
    return result

'''Off-axis scintillation index with strong fluctuations for a tracked beam:
Eq. 43. Calculates off-axis scintillation index for a tracked beam when the beam
launch aperture is larger than the Fried parameter (strong fluctuations), as a
function of path length and radial distance from the center of the beam.'''
def si_on_axis_tc(h_0, h, zeta, lamda, w_0):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    result = si_on_axis_rytov(h_0, h, zeta, lamda, w_0) + 5.95 * (h - h_0) ** 2 * sec2(zeta)\
             * (2 * w_0 / r_0) ** (5 / 3) * (alpha_petc(h, zeta, lamda, w_0) / w_r(h, zeta, lamda, w_0)) ** 2
    return result

'''Pointing error variance with path length:
alpha_pe = sigma_pe / path length, where sigma_pe is the pointing error (eq. 30).
Calculates the pointing error as a ratio of the path length for a collimated beam.
Angle is in radians.'''
def alpha_pe(h_0, h, zeta, lamda, w_0):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    c_r2 = 3.86**2
    result = 0.54*(h-h_0)**2*sec2(zeta)\
             * (lamda*l_path(h,zeta)/(2*w_0))**2 * (2*w_0/r_0)**(5/3)\
             * (1 - ((c_r2 * w_0 ** 2 / r_0 ** 2) / (1 + c_r2 * w_0 ** 2 / r_0 ** 2)) ** (1 / 6))
    return np.sqrt(result)/l_path(h,zeta)**2

'''Pointing error variance for tracked beam with path length:
alpha_petc = sigma_petc / path length, where sigma_petc is the pointing error (eq. 40).
Calculates the pointing error as a ratio of the path length for a tracked collimated beam.
Angle is in radians.'''
def alpha_petc(h, zeta, lamda, w_0):
    k = 2 * np.pi / lamda
    r_0 = calc_zenith_correction(zeta) * fried_param(k, h)
    tz = np.sqrt(0.32*(lamda/(2*w_0))**2*(2*w_0/r_0)**(5/3))
    c_r2 = 3.86 ** 2
    result = (np.sqrt(r_0)-l_path(h, zeta)*tz)**2*(1-(c_r2*w_0**2/r_0**2/(1+c_r2*w_0**2/r_0**2))**(1/6))
    return np.sqrt(result)/l_path(h,zeta)

'''Notation from the Andrews' scintillation article:
Eq. 33. Used to make eq. 32 more compact.'''
def mu_u1(h_0, h,zeta,lamda,w_0):
    lam = lampda_r(h, zeta, lamda, w_0)
    the = (1.0-theta_r(h, zeta, lamda, w_0))
    func = lambda x : HV57(x)*(xi(h_0,h,x)**(5/6)*(mp.mpc(lam*xi(h_0,h,x),(1-the*xi(h_0,h,x))))**(5/6)-lam**(5/6)*xi(h_0,h,x)**(5/3))
    result = mp.quad(func,[h_0,h])
    return result.real

'''Normalized distance variable:'''
def xi(h_0,h,x):
    result = 1 - (x-h_0)/(h-h_0)
    return result

'''Fresnel ratio of beam at transmitter:'''
def lampda_0(h, zeta, lamda, w_0):
    k = 2 * np.pi / lamda
    result = 2 * l_path(h,zeta) / (k * w_0 ** 2)
    return result

'''Fresnel ratio of beam at receiver:'''
def lampda_r(h, zeta, lamda, w_0):
    theta_0 = 1
    result = lampda_0(h,zeta,lamda,w_0)/ (theta_0**2+lampda_0(h,zeta,lamda,w_0)**2)
    return result

'''Beam curvature parameter of the beam at receiver:'''
def theta_r(h, zeta, lamda, w_0):
    theta_0 = 1 # collimated beam
    result = theta_0/(theta_0**2+lampda_0(h,zeta,lamda, w_0)**2)
    return result

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

'''Modulation tools from Digital Communications by J.G. Proakis'''

#TODO: add comments
def qfunc(x):
    return 0.5*special.erfc(x/np.sqrt(2.0))

def marcum_qfunc(a,b):
    func = lambda x : x*mp.exp(-(a**2+x**2)/2)*mp.besseli(0,a*x)
    integral = mp.quad(func,[b,np.inf])
    return float(integral.real)

def ber_bpsk(snr):
    return qfunc(np.sqrt(2*snr))

def ber_qpsk(snr):
    return ber_bpsk(snr)

def ber_mpsk_old(snr,M):
    return 2*qfunc(np.sqrt((2*np.log2(M)*np.sin(np.pi/M)**2)*snr))

def ber_mpsk(snr,M):
    sum = 0
    mx = max(M/4,1)
    n = np.arange(1,int(mx))
    for i in n:
        sum = sum+qfunc(np.sqrt(2*np.log2(M)*snr)*np.sin((2*i-1)*np.pi/M))
    mx = max(np.log2(M),2)
    return 2/mx*sum

def ber_mqam(snr,M):
    a = 1 - 1.0/np.sqrt(M)
    sum = 0
    n = np.arange(1,int(np.sqrt(M)/2))
    for i in n:
        sum = sum+qfunc((2*i-1)*np.sqrt(3*np.log2(M)/(M-1)*snr))
    return 4.0/np.log2(M)*a*sum

def ber_dbpsk(snr):
    return 0.5*np.exp(-snr)

def ber_dqpsk(snr):
    a = np.sqrt(2*snr*(1-np.sqrt(0.5)))
    b = np.sqrt(2*snr*(1+np.sqrt(0.5)))
    result = []
    for i,j in zip(a,b):
        #print(i,j)
        result.append(marcum_qfunc(i,j)-0.5*special.i0(i*j)*np.exp(-(i**2+j**2)/2))
        #print(result[-1])
    return result

def spectral_efficiency(M):
    return np.log2(M)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

'''Plot tools'''
def heatmap2d(arr: np.array):
    plt.imshow(arr,cmap='plasma')
    plt.colorbar()
    plt.show()

def quickPlot(x, y, title="Quick plot", label="-", axx="x", axy="y"):
    f1 = plt.figure(title)
    ax = f1.add_subplot(111)
    ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_ylabel(axy)
    ax.set_xlabel(axx)
    plt.legend()
    plt.show()

def dBhistogram(arr, title="Quick plot", label="-", axx="db", axy='probability density'):
    fig, ax = plt.subplots()
    ax.set_ylim([1e-4, 1])
    ax.set_xlim([-55, -0.001])
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.hist(arr, density=True, bins=70, histtype=u'step', label=label)
    ax.set_title(title)
    ax.set_ylabel(axy)
    ax.set_xlabel(axx)
    plt.legend()
    plt.show()

def hexbin(arrX, arrY, title="Quick plot", label="-", axx="x", axy="y"):
    fig, ax = plt.subplots()
    ax.hexbin(arrX, arrY,cmap='afmhot_r',label=label)
    ax.set_title(title)
    ax.set_ylabel(axy)
    ax.set_xlabel(axx);
    plt.legend()
    plt.show()

