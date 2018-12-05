import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import sys
import functools
import itertools
#import waipy

from cwt import cwt


def wavelet_spectrum(time, counts, dt, t1, t2, plot=True, quiet=False, max_time_scale=None):
    """
    Compute and return the wavelet spectrum

    :param time: a list or a np.array instance containing the time corresponding to each bin in the light curve
    :param counts: a list or a np.array instance containing the counts in each bin
    :param dt: the size of the bin in the light curve
    :param t1: beginning of time interval to use for computation. Of course time.min() <= t1 < time.max()
    :param t2: end of time interval to use for computation. Of course time.min() < t2 <= time.max()
    :param plot: (True or False) whether to produce or not a plot of the spectrum. If False, None will be returned
    instead of the Figure instance
    :param quiet: if True, suppress output (default: False)
    :param max_time_scale: if provided, the spectrum will be computed up to this scale (default: None, i.e., use the
    maximum possible scale)
    :return: (results, fig): a tuple containing a dictionary with the results and the figure
    (a matplotlib.Figure instance)
    """

    counts_copy = np.copy(counts)

    idx = (time >= t1) & (time <= t2)
    counts_copy = counts_copy[idx]

    n_events = np.sum(counts_copy)

    if not quiet:

        print("Number of events: %i" % n_events)
        print("Rate: %s counts/s" % (n_events / (t2 - t1)))

    # Do the Continuous Wavelet transform

    # Default parameters
    s0 = 2 * dt # minimum scale
    dj = 0.125 * 2 # this controls the resolution (number of points)

    if max_time_scale is not None:

        # Compute the corresponding J
        J = int(np.floor(np.log2(max_time_scale / s0 * dj * 2) / dj))

    else:

        J = None

    result = cwt(data=counts_copy, mother='MEXICAN HAT', dt=dt, param=2, s0=s0, dj=dj, J=J)

    # import waipy
    # data_norm = waipy.normalize(counts_copy)
    # alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0, 1]
    # result = waipy.cwt(data=data_norm, mother='DOG', dt=dt, param=2, s0=dt * 2, dj=0.25,
    #                    j1=7 / 0.25, pad=1, lag1=alpha, name='x')
    #
    # result['autocorrelation'] = alpha

    if not quiet:

        print("Lag-1 autocorrelation = {:4.8f}".format(result['autocorrelation']))

    if plot:

        figure, sub = plt.subplots(1,1)

        _ = sub.plot(result['period'], (result['global_ws'] + result['autocorrelation']) / result['scale'], 'o')

        sub.set_xlabel(r"$\delta t$ (s)")
        sub.set_ylabel("Power")

        sub.set_xscale("log")
        sub.set_yscale("log")

        figure.tight_layout()

    else:

        figure = None

    return result, figure


def simulate_flat_poisson_background(rate, dt, t1, t2):

    time = np.arange(t1, t2 + dt, dt)
    lc = np.random.poisson(rate * dt, time.shape[0])

    return time, lc


def worker(i, rate, dt, t1, t2, max_time_scale, results_to_save=('global_ws',)):

    time, lc = simulate_flat_poisson_background(rate, dt, t1, t2)

    try:

        result, _ = wavelet_spectrum(time, lc, dt, t1, t2, plot=False, quiet=True, max_time_scale=max_time_scale)

    except:

        raise

    # Delete everything except what needs to be saved
    keys_to_delete = filter(lambda x: (x not in results_to_save), result.keys())

    map(lambda x: result.pop(x), keys_to_delete)

    # Transform the results in float16 to save memory
    if 'global_ws' in results_to_save:

        result['global_ws'] = np.array(result['global_ws'], np.float16)

    return result


def background_spectrum(rate, dt, t1, t2, n_simulations=1000, plot=True, sig_level=68.0, max_time_scale=None):
    """
    Produce the wavelet spectrum for the background, i.e., a flat signal with Poisson noise with the provided rate.
    Using a simple Monte Carlo simulation, it also produces the confidence region at the requested confidence level.

    NOTE: if you request a very high confidence level, you need to ask for many simulations.

    :param rate: the rate of the background (in counts/s)
    :param dt: the binning of the light curve
    :param t1: start time of the light curve
    :param t2: stop time of the light curve
    :param n_simulations: number of simulations to run to produce the confidence region
    :param plot: whether or not to plot the results (default: True)
    :param sig_level: significance level (default: 68.0, corresponding to 68%)
    :return: (low_bound, median, hi_bound, figure)
    """
    worker_wrapper = functools.partial(worker,
                                       rate=rate, dt=dt, t1=t1, t2=t2,
                                       max_time_scale=max_time_scale,
                                       results_to_save=['global_ws'])

    pool = multiprocessing.Pool()

    all_results = []

    # Get one to get the periods (this is to spare memory)
    one_result = worker(0, rate, dt, t1, t2, max_time_scale, results_to_save=['period', 'scale'])
    periods = np.array(one_result['period'])
    scales = np.array(one_result['scale'])

    try:

        for i, res in enumerate(pool.imap_unordered(worker_wrapper, range(n_simulations), chunksize=100)):
        #for i, res in enumerate(itertools.imap(worker_wrapper, range(n_simulations))):

            sys.stderr.write("\r%i / %i" % (i+1, n_simulations))

            all_results.append(res)

    except:

        raise

    finally:

        pool.close()

    low_bound = np.zeros_like(periods)
    median = np.zeros_like(periods)
    hi_bound = np.zeros_like(periods)

    delta = sig_level / 2.0

    for i, scale in enumerate(scales):

        # Get the value from all simulations at this scale
        values = map(lambda x:x['global_ws'][i] / scale, all_results)

        p16, p50, p84 = np.percentile(values, [50.0 - delta, 50.0, 50.0 + delta])

        low_bound[i] = p16
        median[i] = p50
        hi_bound[i] = p84

    if plot:

        figure, sub = plt.subplots(1, 1)

        _ = sub.fill_between(periods, low_bound, hi_bound, alpha=0.5)
        _ = sub.plot(periods, median, lw=2, color='black')

        sub.set_xlabel(r"$\delta t$ (s)")
        sub.set_ylabel("Power")

        sub.set_xscale("log")
        sub.set_yscale("log")

        figure.tight_layout()

    else:

        figure = None

    return low_bound, median, hi_bound, figure


def plot_spectrum_with_background(spectrum_results, low_bound, median, hi_bound, **kwargs):

    figure, sub = plt.subplots(1, 1, **kwargs)

    _ = sub.plot(spectrum_results['period'],
                 (spectrum_results['global_ws'] + spectrum_results['autocorrelation']) / spectrum_results['scale'],
                 'o')

    _ = sub.fill_between(spectrum_results['period'], low_bound, hi_bound, alpha=0.5)
    _ = sub.plot(spectrum_results['period'], median, lw=2, color='black', linestyle='--')

    sub.set_xlabel(r"$\delta t$ (s)")
    sub.set_ylabel("Power")

    sub.set_xscale("log")
    sub.set_yscale("log")

    figure.tight_layout()

    return figure
