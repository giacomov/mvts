import numpy as np
import pycwt

from power_of_two_utils import is_power_of_2

# waipy.cwt(data=data_norm, mother='DOG', dt=dt, param=2, s0=dt * 2, dj=0.25,
#                        J=7 / 0.25, pad=1, alpha=alpha, name='x')





def cwt(data, dt, mother, param, s0, dj, J=None):

    # Make sure we are dealing with a np.array
    data = np.array(data)

    # Check that data have a power of two size
    N = data.size
    assert is_power_of_2(N), "Sample size for CWT is %s, which is not a power of 2" % N

    # Maximum order
    if J is None:

        J = int(np.floor(np.log2(N * dt / s0) / dj))

    # Normalize and standardize data
    data = (data - data.mean()) / np.sqrt(np.var(data))

    # Compute variance of standardized data
    variance = np.var(data)

    # Autocorrelation (lag-1)
    alpha = np.corrcoef(data[0:-1], data[1:])[0, 1]

    # Setup mother wavelet according to user input

    if mother.upper()=='DOG':

        mother = pycwt.DOG(param)

    elif mother.upper()=='MORLET':

        mother = pycwt.Morlet(param)

    elif mother.upper()=='PAUL':

        mother = pycwt.Paul(param)

    elif mother.upper()=='MEXICAN HAT':

        mother = pycwt.DOG(2)

    else:

        raise ValueError("Wavelet %s is not known. Possible values are: DOG, MORLET, PAUL, MEXICAN HAT")

    # Perform Continuous Wavelet Transform
    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(data, dt, dj, s0, J, mother)

    # Compute power
    power = np.abs(wave)**2

    # Compute periods
    period = [e * mother.flambda() for e in scales]

    # Global normalized power spectrum
    global_ws = variance * (np.sum(power.conj().transpose(), axis=0) / N)

    results = {'autocorrelation': alpha, 'period': period, 'global_ws': global_ws,
               'scale': scales, 'wave': wave, 'freqs': freqs, 'coi': coi}

    return results