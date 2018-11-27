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
    
    Load rejection indices

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

    1. Peak and trough times
    2. Peak and trough sharpness
    3. Sharpness ratio(ShR)
    4. Steepness ratio(StR)
    5.
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
        #remove artifacts
        newRejp=np.where(np.isin(pks['off'][s],rejects['off'][s]))
        peaksharp=np.delete(pksharp['off'][s],newRejp)
        risesteep=np.delete(risteep['off'][s],newRejp)
        newRejt=np.where(np.isin(trs['off'][s],rejects['off'][s]))
        troughsharp=np.delete(trsharp['off'][s],newRejt)
        decaysteep=np.delete(desteep['off'][s],newRejt)
        #calculate ratios
        ShR['off'][s]=np.log10(np.max((np.mean(peaksharp) / np.mean(troughsharp), np.mean(troughsharp) / np.mean(peaksharp)))) 
        PTR['off'][s]=np.mean(peaksharp) / np.mean(troughsharp)  
        StR['off'][s]=np.log10(np.max((np.mean(risesteep)/np.mean(decaysteep),np.mean(decaysteep)/np.mean(risesteep))))
        RDR['off'][s]=np.mean(risesteep) / np.mean(decaysteep) 
    #calculate of on group
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
    #calculate for controls
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


def _lowpass200_all(ecog):
    """
    Apply a 200Hz low-pass filter to all data
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    # Define low pass filter
    def lowpass200(x, Fs=512, Fc=200, Ntaps=250):
        taps = sp.signal.firwin(Ntaps, Fc / (Fs / 2.))
        return np.convolve(taps, x, 'same')

    # Apply low pass filter to all data
    ecoglp = _blankecog()
    for s in range(S):
        ecoglp['B'][s] = lowpass200(ecog['B'][s])
        ecoglp['D'][s] = lowpass200(ecog['D'][s])

    return ecoglp


def _remove_hifreqpeaks_all(ecoglp, order=3):
    """
    Apply notch filters to remove high frequency peaks in all data
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    # Define notch filter method
    def hifreqnotch(x, cf, bw, Fs, order):
        '''
        Notch Filter the time series x with a butterworth with center frequency cf
        and bandwidth bw
        '''
        nyq_rate = Fs / 2.
        f_range = [cf - bw / 2., cf + bw / 2.]
        print(cf)
        Wn = (f_range[0] / nyq_rate, f_range[1] / nyq_rate)
        b, a = sp.signal.butter(order, Wn, 'bandstop')
        return sp.signal.filtfilt(b, a, x)

    def multiplenotch(x, cfs, bws, Fs, order):
        """
        Perform all notch filters for a given piece of data
        """
        Nfilters = len(cfs)
        for f in range(Nfilters):
            x = hifreqnotch(x, cfs[f], bws[f], Fs, order)
        return x

    # Define notch filter for each subject
    hicfPD, hibwPD = _hifreqparams()

    # Apply notch filter to all data
    ecoghf = _blankecog()
    for s in range(S):
        ecoghf['B'][s] = multiplenotch(
            ecoglp['B'][s], hicfPD[s], hibwPD[s], Fs=Fs, order=order)
        ecoghf['D'][s] = multiplenotch(
            ecoglp['D'][s], hicfPD[s], hibwPD[s], Fs=Fs, order=order)

    return ecoghf


def _hifreqparams():
    """
    Return the parameters of the notch filters for the data
    These parameters were obtained by visual inspection of the data
    """
    hicfPD = [[118.8],  # S0
              [164.8, 179.7],  # S1
              [69.1, 117, 138.2, 165.2, 166.3, 186.2, 207],  # S2
              [119.8, 170],  # S3
              [106.9, 213.7],  # S4
              [119.8, 161.7, 166.6, 180, 192.8, 211],  # S5
              [79.5, 113.6, 115, 116.1, 119.8, 125.5, 151, 152.3,
                  153.3, 168, 183.2, 185, 212, 213.8, 215],  # S6
              [119.8, 146.7, 179.7],  # S7
              [127.6, 148.5, 176.1, 214],  # S8
              [119.8, 143, 179.7, 189.2],  # S9
              [54, 79, 118.8, 119.8, 145.5, 172.6, 175.7, 177.5, 179.8],  # S10
              [144.3],  # S11
              [140.9, 179.8, 186.6],  # S12
              [119.8, 147.8, 151.8, 153.6, 179.7],  # S13
              [140.5, 174.3],  # S14
              [155.3, 215],  # S15
              [119.8, 121.6, 129.7, 161.8, 179.7, 194.3],  # S16
              [106, 159, 167.4, 178.4, 189.5, 212.4],  # S17
              [112.1, 128.4],  # S18
              [119.8, 132.6],  # S19
              [168],  # S20
              [119.8, 120.3],  # S21
              [92.5, 112.3, 129.6, 148, 160.8, 167,
               168.5, 176.1, 185, 204]  # S22
              ]
    hibwPD = [[0.5],  # S0
              [1, 0.5],  # S1
              [0.5, 3, 0.5, 0.5, 0.5, 2, 8],  # S2.
              [0.5, 3],  # S3
              [0.5, 0.5],  # S4
              [0.5, 1, 0.5, 4, 0.5, 1],  # S5
              [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1,
                  1, 4, 0.5, 2, 0.5, 0.5, 1],  # S6
              [0.5, 0.5, 0.5],  # S7
              [0.5, 0.5, 0.5, 1],  # S8
              [0.5, 0.5, 0.5, 0.5],  # S9
              [1, 0.5, 0.5, 0.5, 2, 1, 1, 1, .5],  # S10
              [1],  # S11
              [2, 0.5, 1],  # S12
              [0.5, 0.5, 1, 1, 0.5],  # S13
              [0.5, 1],  # S14
              [0.5, 2],  # S15
              [0.5, 1.5, 1, 1, 0.5, 1],  # S16
              [8, 8, 0.5, 0.5, 0.5, 1],  # S17
              [0.5, 0.5],  # S18
              [0.5, 1],  # S19
              [4],  # S20
              [0.5, 0.5],  # S21
              [1, 0.5, 1, 4, 0.5, 3, 0.5, 0.5, 5, 6]  # S22
              ]
    return hicfPD, hibwPD


def normalize_signal_power(ecog):
    Fs, t, S, Sdy, flo, fhi = loadmeta()
    for s in range(S):
        ecog['B'][s] = ecog['B'][s] / np.sqrt(np.sum(ecog['B'][s]**2))
        ecog['D'][s] = ecog['D'][s] / np.sqrt(np.sum(ecog['D'][s]**2))
    return ecog




def measure_pac_TORT(ecog, flo, fhi, Fs=512, Nlo=231, Nhi=240):
    """This function esimates PAC on the PD data
    """
    # Calculate PAC
    import pac
    S = len(ecog['B'])

    pacs = _blankecog(dtype=float)
    for s in range(S):
        pacs['B'][s] = pac.mi_tort(ecog['B'][s], ecog['B'][s], flo, fhi, fs=Fs, filter_kwargslo={
                                  'Ntaps': Nlo}, filter_kwargshi={'Ntaps': Nhi})
        pacs['D'][s] = pac.mi_tort(ecog['D'][s], ecog['D'][s], flo, fhi, fs=Fs, filter_kwargslo={
                                  'Ntaps': Nlo}, filter_kwargshi={'Ntaps': Nhi})

    return pacs


def measure_psd(ecog, Hzmed=1):
    """This function calculates the PSD for all signals
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    from tools.spec import fftmed

    psd = _blankecog(dtype=object)
    for s in range(S):
        f, psd['B'][s] = fftmed(ecog['B'][s], Fs=Fs, Hzmed=Hzmed)
        _, psd['D'][s] = fftmed(ecog['D'][s], Fs=Fs, Hzmed=Hzmed)
    return f, psd


def measure_power(ecog):
    """This function calculates the beta and high gamma power for all signals
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()

    # Calculate PSD
    from tools.spec import fftmed
    Hzmed = 0
    psd = _blankecog(dtype=object)
    for s in range(S):
        f, psd['B'][s] = fftmed(ecog['B'][s], Fs=Fs, Hzmed=Hzmed)
        _, psd['D'][s] = fftmed(ecog['D'][s], Fs=Fs, Hzmed=Hzmed)

    # Calculate beta power
    from tools.spec import calcpow
    bp = _blankecog(dtype=float)
    for s in range(S):
        bp['B'][s] = np.log10(calcpow(f, psd['B'][s], flo))
        bp['D'][s] = np.log10(calcpow(f, psd['D'][s], flo))

    # Calculate high gamma power
    hgp = _blankecog(dtype=float)
    for s in range(S):
        hgp['B'][s] = np.log10(calcpow(f, psd['B'][s], fhi))
        hgp['D'][s] = np.log10(calcpow(f, psd['D'][s], fhi))

    # Calculate total power
    tp = _blankecog(dtype=float)
    for s in range(S):
        tp['B'][s] = np.log10(np.sum(ecog['B'][s]**2))
        tp['D'][s] = np.log10(np.sum(ecog['D'][s]**2))

    return bp, hgp, tp


def measure_rigid():
    # Rigidity data
    #rigidB = np.array([2, 1, 99, 2, 1, 0, 0, 2, 2, 1, 3,
      #                 99, 0, 1, 3, 1, 99, 0, 0, 2, 2, 2, 2])
    #rigidD = np.array([0, 0, 99, 1, 0, 0, 0, 1, 1, 0, 2,
      #                 99, 0, 0, 2, 0, 99, 0, 0, 1, 1, 1, 0])
    rigidB = np.array([14,6,6,8,7,14,13,10,12,10,6,11,10,7,15])#rigid
    rigidD = np.array([12,4,2,3,4,14,14,9,10,5,7,11,9,5,14])#rigid
    #rigidB=np.array([9,2,8,12,1,1,2,2,2,6,2,6,3,3,9])#duration
    #rigidD=np.array([9,2,8,12,1,1,2,2,2,6,2,6,3,3,9])#duration
    #rigidB=np.array([36,23,31,27,21,35,40,35,28,32,26,44,37,33,39])#updrs
    #rigidD=np.array([30,16,24,15,17,35,46,30,24,24,26,39,32,21,39])#updrs
    return rigidB, rigidD #, durationB, durationD, updrsB, updrsD

def calculate_comodulogramPAC(ecog, comodkwargs=None):
    """ Calculate the PAC measures for all signals in PD data using comodulogram method
    """
    Fs, t, S, Sdy, flo, fhi = loadmeta()
    if comodkwargs is None:
        comodkwargs = {}

    cpac = _blankecog(dtype=float)
    for s in range(S):
        cpac['B'][s] = _comodPAC(ecog['B'][s], flo, fhi, Fs, **comodkwargs)
        cpac['D'][s] = _comodPAC(ecog['D'][s], flo, fhi, Fs, **comodkwargs)
    return cpac


def _comodPAC(x, flo, fhi, Fs, dp=2, da=4, w_lo=3, w_hi=3, pac_method='mi_tort'):

    # Calculate comodulogram
    # Filter order was based off presuming deHemptinne 2015 used the default FIR1 filter order
    # using eegfilt:
    # https://sccn.ucsd.edu/svn/software/eeglab/functions/sigprocfunc/eegfilt.m
    from pac import comodulogram
    comod = comodulogram(x, x, flo, fhi, dp, da, fs=Fs, pac_method='mi_tort')
    return np.mean(comod)


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


def morletT(x, f0s, Fs, w=7, s=.5):
    """
    Calculate the time-frequency representation of the signal 'x' over the
    frequencies in 'f0s' using morlet wavelets
    Parameters
    ----------
    x : array
        time series
    f0s : array
        frequency axis
    Fs : float
        Sampling rate
    w : float
        Length of the filter in terms of the number of cycles of the oscillation
        whose frequency is the center of the bandpass filter
    s : float
        Scaling factor
    Returns
    -------
    mwt : 2-D array
        time-frequency representation of signal x
    """
    if w <= 0:
        raise ValueError('Number of cycles in a filter must be a positive number.')

    T = len(x)
    F = len(f0s)
    mwt = np.zeros([F, T], dtype=complex)
    for f in range(F):
        mwt[f] = morletf(x, f0s[f], Fs, w=w, s=s)

    return mwt


def morletf(x, f0, Fs, w=7, s=.5, M=None, norm='sss'):
    """
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
        Length of the filter in terms of the number of cycles of the oscillation
        with frequency f0
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
        raise ValueError('Number of cycles in a filter must be a positive number.')

    if M is None:
        M = w * Fs / f0

    morlet_f = sp.signal.morlet(M, w=w, s=s)
    morlet_f = morlet_f

    if norm == 'sss':
        morlet_f = morlet_f / np.sqrt(np.sum(np.abs(morlet_f)**2))
    elif norm == 'abs':
        morlet_f = morlet_f / np.sum(np.abs(morlet_f))
    else:
        raise ValueError('Not a valid wavelet normalization method.')

    mwt_real = np.convolve(x, np.real(morlet_f), mode='same')
    mwt_imag = np.convolve(x, np.imag(morlet_f), mode='same')

    return mwt_real + 1j*mwt_imag


def simphase(T, flo, w=3, dt=.001, randseed=0, returnwave=False):
    """ Simulate the phase of an oscillation
    The first and last second of the oscillation are simulated and taken out
    in order to avoid edge artifacts in the simulated phase

    Parameters
    ----------
    T : float
        length of time of simulated oscillation
    flo : 2-element array (lo,hi)
        frequency range of simulated oscillation
    dt : float
        time step of simulated oscillation
    returnwave : boolean
        option to return the simulated oscillation
    """
    from tools.spec import bandpass_default
    np.random.seed(randseed)
    whitenoise = np.random.randn(int((T+2)/dt))
    theta, _ = bandpass_default(whitenoise, flo, 1/dt, rmv_edge=False, w=w)

    if returnwave:
        return np.angle(sp.signal.hilbert(theta[int(1/dt):int((T+1)/dt)])), theta[int(1/dt):int((T+1)/dt)]
    else:
        return np.angle(sp.signal.hilbert(theta[int(1/dt):int((T+1)/dt)]))


def simfiltonef(T, f_range, Fs, N, samp_buffer=10000):
    """ Simulate a band-pass filtered signal with brown noise
    Input suggestions: f_range=(2,None), Fs=1000, N=1000

    Parameters
    ----------
    T : float
        length of time of simulated oscillation
    Fs : float
        oscillation sampling rate
    f_range : 2-element array (lo,hi)
        frequency range of simulated data
        if None: do not filter
    N : int
        order of filter
    """

    if f_range is None:
        # Do not filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs))
        return brownN
    elif f_range[1] is None:
        # High pass filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs+N*2))
        # Filter
        nyq = Fs / 2.
        if N % 2 == 0:
            print('NOTE: Increased high-pass filter order by 1 in order to be odd')
            N += 1

        taps = sp.signal.firwin(N, f_range[0] / nyq, pass_zero=False)
        brownNf = sp.signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]

    else:
        # Bandpass filter
        # Generate 1/f^2 noise
        brownN = simbrown(int(T*Fs+N*2))
        # Filter
        nyq = Fs / 2.
        taps = sp.signal.firwin(N, np.array(f_range) / nyq, pass_zero=False)
        brownNf = sp.signal.filtfilt(taps, [1], brownN)
        return brownNf[N:-N]


def simbrown(N):
    """Simulate a brown noise signal (power law distribution 1/f^2)
    with N samples"""
    wn = np.random.randn(N)
    return np.cumsum(wn)

