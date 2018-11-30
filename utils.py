"""
util.py
Some functions for analyzing data in this repo
"""

from __future__ import division
import numpy as np
import scipy as sp
from scipy import io
from scipy import signal
import os
#import pacpy


def loadmeta():
    """Load meta data for analysis of PD data.

    Returns
    -------
    Fs : float
        Sampling rate (Hz)
    t : numpy array
        time array corresponding to the eeg signals
    S : int
        Number of PD patients
    Sc : int
        Number of control subjects
    flo : 2-element tuple
        frequency limits of the beta range (Hz)
    fhi : 2-element tuple
        frequency limits for the high gamma range (Hz)
    """

    Fs = 512.  # Sampling rate (Hz)
    t = np.arange(0, 30, 1 / Fs)  # Time series (seconds)
    S = 30
    Sc = 32
    flo = (13,30)
    fhi = (50, 150)
    return Fs, t, S, Sc, flo, fhi


def _blankeeg(dtype=object):
    Fs, t, S, Sc, flo, fhi = loadmeta()
    eeg = {}
    eeg['off'] = np.zeros(S, dtype=dtype)
    eeg['on'] = np.zeros(S, dtype=dtype)
    eeg['C'] = np.zeros(Sc, dtype=dtype)
    return eeg

def loadPD(filepath='AVG_motor.mat',filepathrej='rejectscombined.mat'):
    '''
    Load the data after following preprocessing:
    1. Average referenced
    
    Load rejection indices:
    1. Each index in an array

    Parameters
    ----------
    filepath : string
        path to averaged referenced data
    filepathrej : string
        path to rejection indices
        
    Returns
    -------
    eeg : dict
        Pre-processed voltage traces
        'off' : subject-by-time array for PD patients OFF medication
        'on' : subject-by-time array for PD patients ON medication        
        'C' : subject-by-time array for control subjects
        
    rejects : dict
        rejection indices including muscle artifacts
        'off' : rejection indices for PD patients OFF medication
        'on' : rejection indices for ON medication        
        'C' : rejection indices for control subjects

    '''
    data = io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    eeg = _blankeeg()
    eeg['off'] = data['B'] 
    eeg['on'] = data['D']
    eeg['C'] = data['C']
    
    rejdata = io.loadmat(filepathrej, struct_as_record=False, squeeze_me=True)
    rejects = _blankeeg()    
    rejects['off'] = rejdata['B'] 
    rejects['on'] = rejdata['D']
    rejects['C'] = rejdata['C']
    
    return eeg,rejects


def measure_shape(eeg, rejects, boundaryS=100, ampPC=0, widthS=3, esrmethod='aggregate'):
    """This function calculates the shape measures calculated for analysis
    of the PD data set

    1. Peak and trough times(pks,trs)
    2. Peak and trough sharpness(pksharp,trsharp)
    3. Rise and decay steepnes(risteep,desteep)
    3. Sharpness ratio(ShR)
    4. Steepness ratio(StR)
    5. Peak-to-trough ratio(PTR)
    6. Rise-to-decay ratio(RDR)
    """
    Fs, t, S, Sc, flo, fhi = loadmeta()
    
    from shape import findpt, ex_sharp, esr, rd_steep, rdsr    
    pks = _blankeeg()
    trs = _blankeeg()
    pksharp = _blankeeg()
    trsharp = _blankeeg()
    risteep = _blankeeg()
    desteep = _blankeeg()
    ShR=_blankeeg(dtype=float)
    StR=_blankeeg(dtype=float)
    PTR=_blankeeg(dtype=float)
    RDR=_blankeeg(dtype=float)
    
    #calculate for off group
    for s in range(S):
        pks['off'][s], trs['off'][s] = findpt(eeg['off'][s],rejects['off'][s], flo, Fs=Fs, boundary=boundaryS)
        pksharp['off'][s] = ex_sharp(eeg['off'][s], pks['off'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        trsharp['off'][s] = ex_sharp(eeg['off'][s], trs['off'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        risteep['off'][s], desteep['off'][s] = rd_steep(eeg['off'][s], pks['off'][s], trs['off'][s])
        #remove artifact regions from each metric dict
        newRejp=np.where(np.isin(pks['off'][s],rejects['off'][s]))
        peaksharp=np.delete(pksharp['off'][s],newRejp)
        risesteep=np.delete(risteep['off'][s],newRejp)
        newRejt=np.where(np.isin(trs['off'][s],rejects['off'][s]))
        troughsharp=np.delete(trsharp['off'][s],newRejt)
        decaysteep=np.delete(desteep['off'][s],newRejt)
        #calculate ratios with rejection regions removed
        ShR['off'][s]=np.log10(np.max((np.mean(peaksharp) / np.mean(troughsharp), np.mean(troughsharp) / np.mean(peaksharp)))) 
        PTR['off'][s]=np.mean(peaksharp) / np.mean(troughsharp)  
        StR['off'][s]=np.log10(np.max((np.mean(risesteep)/np.mean(decaysteep),np.mean(decaysteep)/np.mean(risesteep))))
        RDR['off'][s]=np.mean(risesteep) / np.mean(decaysteep) 
    #calculate same for ON group
    
    for s in range(S):
        pks['on'][s], trs['on'][s] = findpt(eeg['on'][s],rejects['on'][s], flo, Fs=Fs, boundary=boundaryS)
        pksharp['on'][s] = ex_sharp(eeg['on'][s], pks['on'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        trsharp['on'][s] = ex_sharp(eeg['on'][s], trs['on'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        risteep['on'][s], desteep['on'][s] = rd_steep(eeg['on'][s], pks['on'][s], trs['on'][s])
        #remove artifacts
        newRejp=np.where(np.isin(pks['on'][s],rejects['on'][s]))
        peaksharp=np.delete(pksharp['on'][s],newRejp)
        risesteep=np.delete(risteep['on'][s],newRejp)
        newRejt=np.where(np.isin(trs['on'][s],rejects['on'][s]))
        troughsharp=np.delete(trsharp['on'][s],newRejt)
        decaysteep=np.delete(desteep['on'][s],newRejt)
        #calculate ratios
        ShR['on'][s]=np.log10(np.max((np.mean(peaksharp) / np.mean(troughsharp),np.mean(troughsharp) / np.mean(peaksharp)))) 
        PTR['on'][s]=np.mean(peaksharp) / np.mean(troughsharp)  
        StR['on'][s]=np.log10(np.max((np.mean(risesteep)/np.mean(decaysteep),np.mean(decaysteep)/np.mean(risesteep))))
        RDR['on'][s]=np.mean(risesteep) / np.mean(decaysteep) 
    #and for control group
    for s in range(Sc):
        pks['C'][s], trs['C'][s] = findpt(eeg['C'][s],rejects['C'][s], flo, Fs=Fs, boundary=boundaryS)
        pksharp['C'][s] = ex_sharp(eeg['C'][s], pks['C'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        trsharp['C'][s] = ex_sharp(eeg['C'][s], trs['C'][s], widthS, ampPC=ampPC, Fs=Fs, fosc=flo)
        risteep['C'][s], desteep['C'][s] = rd_steep(eeg['C'][s], pks['C'][s], trs['C'][s])
        #remove artifacts
        newRejp=np.where(np.isin(pks['C'][s],rejects['C'][s]))
        peaksharp=np.delete(pksharp['C'][s],newRejp)
        risesteep=np.delete(risteep['C'][s],newRejp)
        newRejt=np.where(np.isin(trs['C'][s],rejects['C'][s]))
        troughsharp=np.delete(trsharp['C'][s],newRejt)
        decaysteep=np.delete(desteep['C'][s],newRejt)
        #calculate ratios
        ShR['C'][s]=np.log10(np.max((np.mean(peaksharp) / np.mean(troughsharp), np.mean(troughsharp) / np.mean(peaksharp)))) 
        PTR['C'][s]=np.mean(peaksharp) / np.mean(troughsharp)  
        StR['C'][s]=np.log10(np.max((np.mean(risesteep)/np.mean(decaysteep), np.mean(decaysteep)/np.mean(risesteep))))
        RDR['C'][s]=np.mean(risesteep) / np.mean(decaysteep) 

    return pks,trs,ShR,PTR,StR,RDR


def measure_pac(eeg, rejects, flo, fhi, Fs=512, Nlo=231, Nhi=240):
    """This function esimates PAC on the PD data
    """
    # Calculate PAC
    import pac
    Fs, t, S, Sc, flo, fhi = loadmeta()

    pacs = _blankeeg(dtype=float)
    for s in range(S):
        pacs['off'][s] = pac.ozkurt(eeg['off'][s], eeg['off'][s],rejects['off'][s], flo, fhi, fs=Fs, filter_kwargslo={'Ntaps': Nlo}, filter_kwargshi={'Ntaps': Nhi})
        pacs['on'][s] = pac.ozkurt(eeg['on'][s], eeg['on'][s],rejects['on'][s], flo, fhi, fs=Fs, filter_kwargslo={'Ntaps': Nlo}, filter_kwargshi={'Ntaps': Nhi})
        
    for s in range(Sc):
        pacs['C'][s] = pac.ozkurt(eeg['C'][s], eeg['C'][s],rejects['C'][s], flo, fhi, fs=Fs, filter_kwargslo={'Ntaps': Nlo}, filter_kwargshi={'Ntaps': Nhi})

    return pacs

def metricsegment(eeg,rejects,group,subject,sWindow,sSlide):
    '''Calculates metric window segments for one subject
    Parameters
    ----------
    eeg : numpy array
        path to averaged referenced data
    rejects : numpy array
        path to rejection indices
    group : string
        path to rejection indices
    subject : int
        patient/subject number
    sWindow : int
        window size in secs
    SSilde : int
        slide size in secs
    Returns
    -------
    ShRseg : numpy array
        sharpness ratio values for segments of trace
    StRseg : numpy array
        steepness ratio values for segments of trace
    PACseg : numpy array
        PAC values for segments of trace
    '''
    from pac import firf
    from shape import findpt, ex_sharp, esr, rd_steep, rdsr

    Fs, t, S, Sdy, flo, fhi = loadmeta()
    pacseg={}
    
    windowsize=int(Fs*sWindow)
    slidesize=int(Fs*sSlide)
    lo=firf(eeg[group][subject], flo, Fs, rmvedge=False,Ntaps=231)
    hi=firf(eeg[group][subject], fhi, Fs, rmvedge=False,Ntaps=240)
    lofilt=np.delete(lo,rejects[group][subject])
    hifilt=np.delete(hi,rejects[group][subject])
    n=int(len(lofilt)-windowsize)
    #PAC
    for s in range(0,n,slidesize):
        lo_ = np.angle(sp.signal.hilbert(lofilt[s:windowsize+s]))
        hi_ = np.abs(sp.signal.hilbert(hifilt[s:windowsize+s]))
        pacseg[s] = np.abs(np.sum(hi_ * np.exp(1j * lo_))) / (np.sqrt(len(lo_)) * np.sqrt(np.sum(hi_**2)))
   
    
    pksw={}
    trsw={}
    ShRseg={}
    riseseg={}
    decayseg={}
    StRseg={}
    
    eeg_ = np.delete(eeg[group][subject], rejects[group][subject])
    rejects0=0
    n=len(eeg_)-windowsize
    
    #StR and ShR
    for s in range(0,n,slidesize):
        window0=eeg_[s:windowsize+s]
        pksw[s], trsw[s] = findpt(window0,rejects0, flo, Fs=Fs)
        ps=np.mean(pksw[s])
        ts=np.mean(trsw[s])
        #ShRseg[s]=np.log10(np.max((ps/ts,ts/ps))) 
        #ShRseg[s] = np.log10(esr(window0, pksw[s], trsw[s], widthS=3))
        ShRseg[s] = np.log10(esr(window0, pksw[s], trsw[s], widthS=3, esrmethod='aggregate'))
        riseseg[s], decayseg[s] = rd_steep(window0,pksw[s],trsw[s])
        rs=np.mean(riseseg[s])
        ds=np.mean(decayseg[s])
        StRseg[s]=np.log10(np.max((rs/ds,ds/rs)))
    
    ShRseg=np.array(list(ShRseg.values())).flatten()
    pacseg=np.array(list(pacseg.values())).flatten()
    StRseg=np.array(list(StRseg.values())).flatten()
    
    return pacseg,ShRseg,StRseg

def firf(x,rejects, f_range, fs=512, w=3, rmvedge=True):
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
    taps = sp.signal.firwin(Ntaps, np.array(f_range) / nyq, pass_zero=False)
    x_filt = sp.signal.filtfilt(taps, [1], x)

    if any(np.isnan(x_filt)):
        raise RuntimeError(
            'Filtered signal contains nans. Adjust filter parameters.')

    # Remove edge artifacts
    if rmvedge:
        return _remove_edge(x_filt, Ntaps)
    else:
        return np.delete(x_filt, rejects)

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




