"""
shape.py
Tools for quantifying the waveform shape of oscillations
"""

from __future__ import division
import numpy as np
import pac
#import pacpy #wasn't here before
from scipy import io
from scipy.signal import firwin2, firwin
from scipy.signal import filtfilt


"""
def filtandremove(x,rejects, f_osc, Fs=512., w=3, boundary=0):
        # Filter in narrow band
    from pacpy.filt import firf #in filt.py#pacpy.filt
    xn = firf(x, f_osc, Fs, w=w, rmvedge=False)

    #remove rejection indices
    xn = np.delete(xn, rejects)
    
    return xn
"""
def findpt(x, rejects, f_osc, Fs=512., w=3, boundary=0):
    """
    Calculate peaks and troughs over time series

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    f_osc : (low, high), Hz
        frequency range for narrowband signal of interest, used to find
        zerocrossings of the oscillation
    Fs : float
        The sampling rate (default = 1000Hz)
    w : float
        Number of cycles for the filter order of the band-pass filter
    boundary : int
        distance from edge of recording that an extrema must be in order to be
        accepted (in number of samples)

    Returns
    -------
    Ps : array-like 1d
        indices at which oscillatory peaks occur in the input signal x
    Ts : array-like 1d
        indices at which oscillatory troughs occur in the input signal 
        """
    # Filter in narrow band
    #from pacpy.filt import firf
    #from utils import firf
    xn = firf(x, f_osc, Fs, w=w, rmvedge=False)

    # Find zero crosses
    def fzerofall(data):
        pos = data > 0
        return (pos[:-1] & ~pos[1:]).nonzero()[0]
    #[:-1]=denotes all but last element in pos
    #~=bits of pos inverted

    def fzerorise(data):
        pos = data < 0
        return (pos[:-1] & ~pos[1:]).nonzero()[0]

    zeroriseN = fzerorise(xn)
    zerofallN = fzerofall(xn)

    # Calculate # peaks and troughs
    if zeroriseN[-1] > zerofallN[-1]:
        P = len(zeroriseN) - 1
        T = len(zerofallN)
    else:
        P = len(zeroriseN)
        T = len(zerofallN) - 1
    if T-P==2:
        T=T-2
        
    # Calculate peak samples
    Ps = np.zeros(P, dtype=int)
    for p in range(P):
        # Calculate the sample range between the most recent zero rise
        # and the next zero fall
        mrzerorise = zeroriseN[p]
        nfzerofall = zerofallN[zerofallN > mrzerorise][0]
        Ps[p] = np.argmax(x[mrzerorise:nfzerofall]) + mrzerorise

    # Calculate trough samples
    Ts = np.zeros(T, dtype=int)
    for tr in range(T):
        # Calculate the sample range between the most recent zero fall
        # and the next zero rise
        mrzerofall = zerofallN[tr]
        nfzerorise = zeroriseN[zeroriseN > mrzerofall][0]
        Ts[tr] = np.argmin(x[mrzerofall:nfzerorise]) + mrzerofall

    if boundary > 0:
        Ps = _removeboundaryextrema(x, Ps, boundary)
        Ts = _removeboundaryextrema(x, Ts, boundary)
        
    return Ps, Ts


def _removeboundaryextrema(x, Es, boundaryS):
    """
    Remove extrema close to the boundary of the recording

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Es : array-like 1d
        time points of oscillatory peaks or troughs
    boundaryS : int
        Number of samples around the boundary to reject extrema

    Returns
    -------
    newEs : array-like 1d
        extremas that are not too close to boundary

    """

    # Calculate number of samples
    nS = len(x)

    # Reject extrema too close to boundary
    SampLims = (boundaryS, nS - boundaryS)
    E = len(Es)
    todelete = []
    for e in range(E):
        if np.logical_or(Es[e] < SampLims[0], Es[e] > SampLims[1]):
            todelete = np.append(todelete, e)

    newEs = np.delete(Es, todelete)

    return newEs


def wfpha(x, Ps, Ts):
    """
    Use peaks and troughs calculated with findpt to calculate an instantaneous
    phase estimate over time

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs

    Returns
    -------
    pha : array-like 1d
        instantaneous phase
    """

    # Initialize phase array
    L = len(x)
    pha = np.empty(L)
    pha[:] = np.NAN

    pha[Ps] = 0
    pha[Ts] = -np.pi

    # Interpolate to find all phases
    marks = np.logical_not(np.isnan(pha))
    t = np.arange(L)
    marksT = t[marks]
    M = len(marksT)
    for m in range(M - 1):
        idx1 = marksT[m]
        idx2 = marksT[m + 1]

        val1 = pha[idx1]
        val2 = pha[idx2]
        if val2 <= val1:
            val2 = val2 + 2 * np.pi

        phatemp = np.linspace(val1, val2, idx2 - idx1 + 1)
        pha[idx1:idx2] = phatemp[:-1]

    # Interpolate the boundaries with the same rate of change as the adjacent
    # sections
    idx = np.where(np.logical_not(np.isnan(pha)))[0][0]
    val = pha[idx]
    dval = pha[idx + 1] - val
    startval = val - dval * idx
    # .5 for nonambiguity in arange length
    pha[:idx] = np.arange(startval, val - dval * .5, dval)

    idx = np.where(np.logical_not(np.isnan(pha)))[0][-1]
    val = pha[idx]
    dval = val - pha[idx - 1]
    dval = np.angle(np.exp(1j * dval))  # Trestrict dval to between -pi and pi
    # .5 for nonambiguity in arange length
    endval = val + dval * (len(pha) - idx - .5)
    pha[idx:] = np.arange(val, endval, dval)

    # Restrict phase between -pi and pi
    pha = np.angle(np.exp(1j * pha))

    return pha


def _edgeadd_paseries(amp, fosc, Fs, w=3):
    """
    Undo the removal of edge artifacts done by pacpy in order to align
    the extrema with their amplitudes
    """
    Ntaps = np.int(np.floor(w * Fs / fosc[0]))
    amp2 = np.zeros(len(amp) + 2 * Ntaps)
    amp2[Ntaps:-Ntaps] = amp
    return amp2


def ex_sharp(x, Es, widthS, ampPC=0, Fs=512, fosc=(13, 30), method='diff'):
    """
    Calculate the sharpness of extrema

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Es : array-like 1d
        time points of oscillatory peaks or troughs
    widthS : int
        Number of samples in each direction around extrema to use for sharpness estimation
    ampPC : float (0 to 100)
        The percentile threshold of beta (or other oscillation) amplitude
        for which an extrema needs to be included in the analysis
    Fs : float
        Sampling rate
    fosc : (low, high), Hz
        The frequency range of the oscillation identified by the extrema

    Returns
    -------
    sharps : array-like 1d
        sharpness of each extrema is Es

    """
    E = len(Es)#Es is pks or trs, function is called twice, once for each
    sharps = np.zeros(E)#returns array filled with zeroes that correspond to number of peaks
    for e in range(E):
        if method == 'deriv':
            Edata = x[Es[e] - widthS: Es[e] + widthS + 1]
            sharps[e] = np.mean(np.abs(np.diff(Edata)))
        elif method == 'diff':#method used is diff
            sharps[e] = np.mean((x[Es[e]] - x[Es[e] - widthS], x[Es[e]] - x[Es[e] + widthS]))
            #mean of (Voltage at peak/trough-Voltage at 5 samples before) and (Voltage at peak/trough + Voltage at 5 samples before)(Figure 1.B)
    sharps = np.abs(sharps)#absolute value

    return _ampthresh(ampPC, x, fosc, Fs, Es, sharps)

def _ampthresh(ampTH, x, fosc, Fs, Es, metric):
    """
    Restrict data to the time points at which the extrema
    """
    #since ampTH = 0, nothing happens
    if ampTH > 0:
        from pacpy.pac import pa_series
        _, bamp = pa_series(x, x, fosc, fosc, fs=Fs)
        bamp = _edgeadd_paseries(bamp, fosc, Fs)
        bamp = bamp[Es]
        bampTH = np.percentile(bamp, ampTH)
        metric = metric[bamp >= bampTH]

    return metric


def rd_steep(x, Ps, Ts):
    """
    Calculate the max steepness of rises and decays

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs

    Returns
    -------
    risesteep : array-like 1d
        max steepness in each period for rise
    decaysteep : array-like 1d
        max steepness in each period for decay

    """

    #if len(Ts)-len(Ps)==2:
        #Ts=Ts[:-1]
    # Calculate the max Rise steepness (after trough)
    if Ps[0] < Ts[0]:
        riseadj = 1
    else:
        riseadj = 0

    T = len(Ts) - 1
    risesteep = np.zeros(T)
    for t in range(T):
        rise = x[Ts[t]:Ps[t + riseadj] + 1]
        risesteep[t] = np.max(np.diff(rise))#error here

    P = len(Ps) - 1
    decaysteep = np.zeros(P)
    for p in range(P):
        decay = x[Ps[p]:Ts[p - riseadj + 1] + 1]
        decaysteep[p] = -np.min(np.diff(decay))

    return risesteep, decaysteep


def rd_steepidx(x, Ps, Ts):
    """
    Calculate the indices of max steepness of rises and decays

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of oscillatory troughs

    Returns
    -------
    risesteep : array-like 1d
        indices of max steepness in each period for rise
    decaysteep : array-like 1d
        indices of max steepness in each period for decay

    """
    if Ps[0] < Ts[0]:
        riseadj = 1
    else:
        riseadj = 0

    T = len(Ts) - 1
    risesteep = np.zeros(T)
    for t in range(T):
        rise = x[Ts[t]:Ps[t + riseadj] + 1]
        risesteep[t] = Ts[t] + np.argmax(np.diff(rise))

    P = len(Ps) - 1
    decaysteep = np.zeros(P)
    for p in range(P):
        decay = x[Ps[p]:Ts[p - riseadj + 1] + 1]
        decaysteep[p] = Ps[p] + np.argmin(np.diff(decay))

    return risesteep, decaysteep


def rdsr(Rsteep, Dsteep):
    return np.max((np.mean(Rsteep) / np.mean(Dsteep), np.mean(Dsteep) / np.mean(Rsteep)))


def esr(x, Ps, Ts, widthS, ampPC=0, Fs=512, fosc=(13, 30),
        pthent=True, esrmethod='adjacent'):
    """Calculate extrema sharpness ratio: the peak/trough sharpness ratio
    but fixed to be above 1.
    Pairs are peaks and subsequent troughs

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of osillatory troughs
    widthS : int
        Number of samples in each direction around extrema to use for sharpness estimation
    ampPC : float (0 to 100)
        The percentile threshold of beta (or other oscillation) amplitude
        for which an extrema needs to be included in the analysis
    Fs : float
        Sampling rate
    fosc : (low, high), Hz
        The frequency range of the oscillation identified by the extrema
    pthent : bool
        if True: a period is defined as a peak and subsequent trough
        if False: a period is defined as a trough and subsequent peak
    esrmethod : string ('adjacent' or 'aggregate)


    Returns
    -------
    esr : array-like 1d
        extrema sharpness ratio for each period
    """

    if esrmethod == 'adjacent':
        PTr = _PTrsharp(x, Ps, Ts, widthS, ampPC=ampPC, Fs=Fs, fosc=fosc,
                        pthent=pthent)
        return np.max(np.vstack((PTr, 1 / PTr)), axis=0)
    elif esrmethod == 'aggregate':
        psharp = np.mean(ex_sharp(x, Ps, widthS, Fs=Fs, ampPC=ampPC, fosc=fosc))#np.mean
        tsharp = np.mean(ex_sharp(x, Ts, widthS, Fs=Fs, ampPC=ampPC, fosc=fosc))
        #if len(psharp)>len(tsharp):
         #   psharp=psharp[:-1]
        #if len(psharp)<len(tsharp):
         #   tsharp=tsharp[:-1]
        esr = np.max((psharp / tsharp, tsharp / psharp))#finding the maximum sharpness ratio
        return esr
    else:
        raise ValueError('Not a valid esrmethod entry')


def _PTrsharp(x, Ps, Ts, widthS, ampPC=0, Fs=1000, fosc=(13, 30),
              pthent=True):
    """Calculate peak-trough sharpness ratio

    Parameters
    ----------
    x : array-like 1d
        voltage time series
    Ps : array-like 1d
        time points of oscillatory peaks
    Ts : array-like 1d
        time points of osillatory troughs
    widthS : int
        Number of samples in each direction around extrema to use for sharpness estimation
    ampPC : float (0 to 100)
        The percentile threshold of beta (or other oscillation) amplitude
        for which an extrema needs to be included in the analysis
    Fs : float
        Sampling rate
    fosc : (low, high), Hz
        The frequency range of the oscillation identified by the extrema
    pthent : bool
        if True: a period is defined as a peak and subsequent trough
        if False: a period is defined as a trough and subsequent peak


    Returns
    -------
    ptr : array-like 1d
        peak-trough sharpness ratio for each period
    """

    # Calculate sharpness of peaks and troughs
    Psharp = ex_sharp(x, Ps, widthS)
    Tsharp = ex_sharp(x, Ts, widthS)

    # Align peak and trough arrays to one another
    if pthent:
        if Ts[0] < Ps[0]:
            Tsharp = Tsharp[1:]
            Ts = Ts[1:]
        if len(Psharp) == len(Tsharp) + 1:
            Psharp = Psharp[:-1]
            Ps = Ps[:-1]
    else:
        if Ps[0] < Ts[0]:
            Psharp = Psharp[1:]
            Ps = Ps[1:]
        if len(Tsharp) == len(Psharp) + 1:
            Tsharp = Tsharp[:-1]
            Ts = Ts[:-1]

    ptr = Psharp / Tsharp

    # Only look at sharpness for sufficiently high oscillation amplitude
    if pthent:
        Es = Ps
    else:
        Es = Ts
    return _ampthresh(ampPC, x, fosc, Fs, Es, ptr)

def firf(x, f_range, fs=512, w=4, rmvedge = True):
    """
    Filter signal with an FIR filter
    *Like fir1 in MATLAB

    x : array-like, 1d
        Time series to filter
    f_range : (low, high), Hz
        Cutoff frequencies of bandpass filter
    fs : float, Hz
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles 
        of the oscillation whose frequency is the low cutoff of the 
        bandpass filter

    Returns
    -------
    x_filt : array-like, 1d
        Filtered time series
    """

    if w <= 0:
        raise ValueError(
            'Number of cycles in a filter must be a positive number.')

    nyq = np.float(fs / 2)
    if np.any(np.array(f_range) > nyq):
        raise ValueError('Filter frequencies must be below nyquist rate.')

    if np.any(np.array(f_range) < 0):
        raise ValueError('Filter frequencies must be positive.')

    Ntaps = np.floor(w * fs / f_range[0])
    if len(x) < Ntaps:
        raise RuntimeError(
            'Length of filter is loger than data. '
            'Provide more data or a shorter filter.')

    # Perform filtering
    tapsx = firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)#####
    x_filt = filtfilt(tapsx, [1], x)

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
