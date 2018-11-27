# -*- coding: utf-8 -*-
"""
pac.py
Algorithms for estimating phase-amplitude coupling
"""
import numpy as np
import scipy as sp


def ozkurt(lo, hi, rejects, f_lo, f_hi, fs=512,
           filterfn=None, filter_kwargslo=None, filter_kwargshi=None,removeart=True):
    """
    Calculate PAC using the method defined in Ozkurt & Schnitzler, 2011

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    removeart : 
        True if remove artefacts

        False activates 'EXPERT MODE'.
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING!
        - In expert mode the user needs to filter the data AND apply the
        hilbert transform.
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`

    Returns
    -------
    pac : scalar
      PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import ozkurt
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> ozkurt(lo, hi, (4,8), (80,150)) # Calculate PAC
    """

    lo, hi = pa_series(lo, hi, f_lo, f_hi, fs=fs,
                       filterfn=filterfn, filter_kwargslo=filter_kwargslo, filter_kwargshi=filter_kwargshi)
    
    if removeart is True:
        lo = np.delete(lo, rejects)
        hi = np.delete(hi, rejects)
    
    # Calculate PAC
    pac = np.abs(np.sum(hi * np.exp(1j * lo))) / \
        (np.sqrt(len(lo)) * np.sqrt(np.sum(hi**2)))
        
    return pac


def mi_tort(lo, hi, f_lo, f_hi, fs=512,
            Nbins=20, filterfn=None, filter_kwargslo=None, filter_kwargshi=None):
    """
    Calculate PAC using the modulation index method from prefiltered
    signals

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to ue as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering ranges
    f_hi : (low, high), Hz
        The low frequency filtering range=
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : functional
        The filtering function, `filterfn(x, f_range, filter_kwargs)`

        False activates 'EXPERT MODE'.
        - DO NOT USE THIS FLAG UNLESS YOU KNOW WHAT YOU ARE DOING!
        - In expert mode the user needs to filter the data AND apply the
        hilbert transform.
        - This requires that 'lo' be the phase time series of the low-bandpass
        filtered signal, and 'hi' be the amplitude time series of the high-
        bandpass of the original signal.
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    Nbins : int
        Number of bins to split up the low frequency oscillation cycle

    Returns
    -------
    pac : scalar
        PAC value

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import mi_tort
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> mi_tort(lo, hi, (4,8), (80,150)) # Calculate PAC
    0.34898478944110811
    """

    # Arg check
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError(
            'Number of bins in the low frequency oscillation cycle'
            'must be an integer >1.')

    lo, hi = pa_series(lo, hi, f_lo, f_hi, fs=fs,
                       filterfn=filterfn, filter_kwargslo=filter_kwargslo, filter_kwargshi=filter_kwargshi)

    # Convert the phase time series from radians to degrees
    phadeg = np.degrees(lo)

    # Calculate PAC
    binsize = 360 / Nbins
    phase_lo = np.arange(-180, 180, binsize)
    mean_amp = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        phaserange = np.logical_and(phadeg >= phase_lo[b],
                                    phadeg < (phase_lo[b] + binsize))
        mean_amp[b] = np.mean(hi[phaserange])

    p_j = np.zeros(len(phase_lo))
    for b in range(len(phase_lo)):
        p_j[b] = mean_amp[b] / sum(mean_amp)

    h = -np.sum(p_j * np.log10(p_j))
    h_max = np.log10(Nbins)
    pac = (h_max - h) / h_max

    return pac


def comodulogram(lo, hi, p_range, a_range, dp, da, fs, pac_method='mi_tort', w=7):
    """
    Calculate PAC for many small frequency bands

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    p_range : (low, high), Hz
        The low frequency filtering range
    a_range : (low, high), Hz
        The high frequency filtering range
    dp : float, Hz
        Step size of the low-frequency filter center frequency for each PAC calculation
    da : float, Hz
        Step size of the high-frequency filter center frequency for each PAC calculation
    fs : float
        The sampling rate (default = 1000Hz)
    pac_method : string
        Method to calculate PAC.
        'mi_tort' - See Tort, 2008
        'plv' - See Penny, 2008
        'glm' - See Penny, 2008
        'mi_canolty' - See Canolty, 2006
        'ozkurt' - See Ozkurt & Schnitzler, 2011
    w : float
        Number of cycles for each morlet filter

    Returns
    -------
    comod : array-like, 2d
        Matrix of phase-amplitude coupling values for each combination of the
        phase frequency bin and the amplitude frequency bin

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import comodulogram
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> comod = comodulogram(lo, hi, (5,25), (75,175), 10, 50) # Calculate PAC
    >>> print comod
    [[ 0.32708628  0.32188585]
     [ 0.3295994   0.32439953]]
    """

    # Arg check
    if dp <= 0:
        raise ValueError('Width of lo frequency range must be positive')
    if da <= 0:
        raise ValueError('Width of hi frequency range must be positive')

    # method check
    method2fun = {'mi_tort': mi_tort, 'ozkurt': ozkurt}
    pac_fun = method2fun.get(pac_method, None)
    if pac_fun is None:
        raise ValueError('PAC method given is invalid.')

    # Calculate palette frequency parameters
    f_phases = np.arange(p_range[0], p_range[1] + dp, dp)
    f_amps = np.arange(a_range[0], a_range[1] + da, da)
    P = len(f_phases)
    A = len(f_amps)

    # Calculate all phase time series
    phaseT = np.zeros(P, dtype=object)
    for p in range(P):
        f_lo = f_phases[p]
        loF = morletf(lo, f_lo, fs, w=w)
        phaseT[p] = np.angle(loF)

    # Calculate all amplitude time series
    ampT = np.zeros(A, dtype=object)
    for a in range(A):
        f_hi = f_amps[a]
        hiF = morletf(hi, f_hi, fs, w=w)
        ampT[a] = np.abs(hiF)

    # Calculate PAC for every combination of P and A
    comod = np.zeros((P, A))
    for p in range(P):
        for a in range(A):
            comod[p, a] = pac_fun(phaseT[p], ampT[a], [],
                                  [], fs=fs, filterfn=False)
    return comod


def pa_series(lo, hi, f_lo, f_hi, fs=512,
              filterfn=None, filter_kwargslo=None, filter_kwargshi=None, hi_phase=False):
    """
    Calculate the phase and amplitude time series

    Parameters
    ----------
    lo : array-like, 1d
        The low frequency time-series to use as the phase component
    hi : array-like, 1d
        The high frequency time-series to use as the amplitude component
    f_lo : (low, high), Hz
        The low frequency filtering range
    f_hi : (low, high), Hz
        The low frequency filtering range
    fs : float
        The sampling rate (default = 1000Hz)
    filterfn : function
        The filtering function, `filterfn(x, f_range, filter_kwargs)`
    filter_kwargs : dict
        Keyword parameters to pass to `filterfn(.)`
    hi_phase : boolean
        Whether to calculate phase of low-frequency component of the high frequency
        time-series amplitude instead of amplitude of the high frequency time-series
        (default = False)

    Returns
    -------
    pha : array-like, 1d
        Time series of phase
    amp : array-like, 1d
        Time series of amplitude (or phase of low frequency component of amplitude if hi_phase=True)

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import pa_series
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> pha, amp = pa_series(lo, hi, (4,8), (80,150))
    >>> print pha
    [ 1.57079633  1.60849544  1.64619455 ...,  1.45769899  1.4953981  1.53309721]
    """

    
  




    # Filter setup
    if filterfn is None:
        filterfn = firf

    if filter_kwargslo is None:
        filter_kwargslo = {}
    if filter_kwargshi is None:
        filter_kwargshi = {}

    # Filter then hilbert
    if filterfn is not False:
        lo = filterfn(lo, f_lo, fs, **filter_kwargslo)
        hi = filterfn(hi, f_hi, fs, **filter_kwargshi)

        lo = np.angle(sp.signal.hilbert(lo))
        hi = np.abs(sp.signal.hilbert(hi))

        # if high frequency should be returned as phase of low-frequency
        # component of the amplitude:
        if hi_phase:
            hi = filterfn(hi, f_lo, fs, **filter_kwargslo)
            hi = np.angle(sp.signal.hilbert(hi))
            
        # Make arrays the same size
        lo, hi = _trim_edges(lo, hi)

    return lo, hi


def pa_dist(pha, amp, Nbins=10):
    """
    Calculate distribution of amplitude over a cycle of phases

    Parameters
    ----------
    pha : array
        Phase time series
    amp : array
        Amplitude time series
    Nbins : int
        Number of phase bins in the distribution,
        uniformly distributed between -pi and pi.

    Returns
    -------
    dist : array
        Average amplitude in each phase bins
    phase_bins : array
        The boundaries to each phase bin. Note the length is 1 + len(dist)

    Usage
    -----
    >>> import numpy as np
    >>> from scipy.signal import hilbert
    >>> from pacpy.pac import pa_series, pa_dist
    >>> t = np.arange(0, 10, .001) # Define time array
    >>> lo = np.sin(t * 2 * np.pi * 6) # Create low frequency carrier
    >>> hi = np.sin(t * 2 * np.pi * 100) # Create modulated oscillation
    >>> hi[np.angle(hilbert(lo)) > -np.pi*.5] = 0 # Clip to 1/4 of cycle
    >>> pha, amp = pa_series(lo, hi, (4,8), (80,150))
    >>> phase_bins, dist = pa_dist(pha, amp)
    >>> print dist
    [  7.21154110e-01   8.04347122e-01   4.49207087e-01   2.08747058e-02
       8.03854240e-05   3.45166617e-05   3.45607343e-05   3.51091029e-05
       7.73644631e-04   1.63514941e-01]
    """
    if np.logical_or(Nbins < 2, Nbins != int(Nbins)):
        raise ValueError(
            'Number of bins in the low frequency oscillation cycle must be an integer >1.')
    if len(pha) != len(amp):
        raise ValueError(
            'Phase and amplitude time series must be of same length.')

    phase_bins = np.linspace(-np.pi, np.pi, int(Nbins + 1))
    dist = np.zeros(int(Nbins))

    for b in range(int(Nbins)):
        t_phase = np.logical_and(pha >= phase_bins[b],
                                 pha < phase_bins[b + 1])
        dist[b] = np.mean(amp[t_phase])

    return phase_bins[:-1], dist


def _trim_edges(lo, hi):
    """
    Remove extra edge artifact from the signal with the shorter filter
    so that its time series is identical to that of the filtered signal
    with a longer filter.
    """

    if len(lo) == len(hi):
        return lo, hi  # Die early if there's nothing to do.
    elif len(lo) < len(hi):
        Ndiff = len(hi) - len(lo)
        if Ndiff % 2 != 0:
            raise ValueError(
                'Difference in filtered signal lengths should be even')
        hi = hi[np.int(Ndiff / 2):np.int(-Ndiff / 2)]
    else:
        Ndiff = len(lo) - len(hi)
        if Ndiff % 2 != 0:
            raise ValueError(
                'Difference in filtered signal lengths should be even')
        lo = lo[np.int(Ndiff / 2):np.int(-Ndiff / 2)]

    return lo, hi


def firmorlet(x, f_range, fs, s=1, norm='abs', rmvedge=True):
    """
    This function applies a morlet filter in which the defined
    cutoff frequencies have attenuation of -3dB. It returns only the
    real component.

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    s : float
        Scaling factor for the morlet wavelet
    norm : string
        Normalization method
        'sss' - divide by the sqrt of the sum of squares of points
        'amp' - divide by the sum of amplitudes divided by 2
    rmvedge : boolean
        Option to remove edge artifacts or keep them in

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series

    Returns
    -------
    x_trans : array
        Complex time series
    """

    # Calculate number of cycles from bandwidth
    bandwidth = f_range[1] - f_range[0]
    f0 = np.mean(f_range)
    w = 1.7 * f0 / bandwidth

    # Caculate filter length
    M = 2 * s * w * fs / f0

    # Design filter
    morlet_f = sp.signal.morlet(M, w=w, s=s)

    # Normalize by sum of squared amplitude
    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    # Apply filter
    x_filt = np.convolve(x, np.real(morlet_f), mode='full')
    x_filt = x_filt[int(M / 2.):int(-M / 2.)]

    # Remove edge artifacts
    if rmvedge:
        return _remove_edge(x_filt, M)
    else:
        return x_filt


def morletf(x, f0, fs, w=3, s=1, M=None, norm='sss'):
    """
    NOTE: This function is not currently ready to be interfaced with pacpy
    This is because the frequency input is not a range, which is a big
    assumption in how the api is currently designed

    Convolve a signal with a complex wavelet
    The real part is the filtered signal
    Taking np.abs() of output gives the analytic amplitude
    Taking np.angle() of output gives the analytic phase

    x : array
        Time series to filter
    f0 : float
        Center frequency of bandpass filter
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of
        cycles of the oscillation with frequency f0
    s : float
        Scaling factor for the morlet wavelet
    M : integer
        Length of the filter. Overrides the f0 and w inputs
    norm : string
        Normalization method
        'sss' - divide by the sqrt of the sum of squares of points
        'amp' - divide by the sum of amplitudes divided by 2

    Returns
    -------
    x_trans : array
        Complex time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    if M is None:
        M = 2 * s * w * fs / f0

    morlet_f = sp.signal.morlet(M, w=w, s=s)

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f)) * 2
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    x_filtR = np.convolve(x, np.real(morlet_f), mode='same')
    x_filtI = np.convolve(x, np.imag(morlet_f), mode='same')

    return x_filtR + 1j * x_filtI


def firf(x, f_range, fs, Ntaps=None, rmvedge=False):
    """
    Filter signal with an FIR filter
    *Like fir1 in MATLAB

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    Ntaps : int
        Filter order (length in samples)
        None defaults to 3 cycle lengths of low cutoff frequency

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    nyq = np.float(fs / 2)
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    # Default filter length is 3 cycle lengths
    if Ntaps is None:
        Ntaps = np.floor(3 * fs / f_range[0])

    if len(x) < Ntaps:
        print(len(x))
        raise RuntimeError(
            'Length of filter is loger than data. '
            'Provide more data or a shorter filter.')

    # Perform filtering
    taps = sp.signal.firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    x_filt = sp.signal.filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    # Remove edge artifacts
    if rmvedge:
        return _remove_edge(x_filt, Ntaps)
    else:
        return x_filt


def _remove_edge(x, N):
    """
    Calculate the number of points to remove for edge artifacts

    x : array
        time series to remove edge artifacts from
    N : int
        length of filter
    """
    N = int(N)
    return x[N:-N]
