import waipy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import sys
import functools


def wavelet_spectrum(time, counts, dt, t1, t2, plot=True, quiet=False):
    """
    Compute and return the wavelet spectrum

    :param time: a list or a np.array instance containing the time corresponding to each bin in the light curve
    :param counts: a list or a np.array instance containing the counts in each bin
    :param dt: the size of the bin in the light curve
    :param t1: beginning of time interval to use for computation. Of course time.min() <= t1 < time.max()
    :param t2: end of time interval to use for computation. Of course time.min() < t2 <= time.max()
    :param plot: (True or False) whether to produce or not a plot of the spectrum. If False, None will be returned
    instead of the Figure instance
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

    # Normalize and withen data
    data_norm = waipy.normalize(counts_copy)

    # Compute alpha, i.e., autocorrelation
    alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0, 1]

    if not quiet:

        print("Lag-1 autocorrelation = {:4.8f}".format(alpha))

    # Do the Continuous Wavelet transform

    result = waipy.cwt(data=data_norm, mother='DOG', dt=dt, param=2, s0=dt * 2, dj=0.25,
                       j1=7 / 0.25, pad=1, lag1=alpha, name='x')

    # Add the autocorrelation:

    result['autocorrelation'] = alpha
    #power = result['global_ws'] * len(result['data']) / np.var(result['data'])

    if plot:

        figure, sub = plt.subplots(1,1)

        _ = sub.plot(result['period'], result['global_ws'] + alpha)

        sub.set_xlabel("Time scale (s)")
        sub.set_ylabel("Power")

        sub.set_xscale("log")

        figure.tight_layout()

    else:

        figure = None

    return result, figure


def simulate_flat_poisson_background(rate, dt, t1, t2):

    time = np.arange(t1, t2, dt)
    lc = np.random.poisson(rate * dt, time.shape[0])

    return time, lc


def worker(i, rate, dt, t1, t2):

    time, lc = simulate_flat_poisson_background(rate, dt, t1, t2)

    try:

        result, _ = wavelet_spectrum(time, lc, dt, t1, t2, plot=False, quiet=True)

    except:

        raise

    return result


def background_spectrum(rate, dt, t1, t2, n_simulations=1000, plot=True, sig_level=68.0):
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
    worker_wrapper = functools.partial(worker, rate=rate, dt=dt, t1=t1, t2=t2)

    pool = multiprocessing.Pool()

    all_results = []

    try:

        for i, res in enumerate(pool.imap_unordered(worker_wrapper, range(n_simulations), chunksize=2)):

            sys.stderr.write("\r%i / %i" % (i+1, n_simulations))

            # Delete everything except 'period' and 'global_ws'
            keys_to_delete = filter(lambda x:(x!='period' and x!='global_ws'), res.keys())

            map(lambda x:res.pop(x), keys_to_delete)

            all_results.append(res)

    except:

        raise

    finally:

        pool.close()

    # Get the periods
    periods = np.array(all_results[0]['period'])

    low_bound = np.zeros_like(periods)
    median = np.zeros_like(periods)
    hi_bound = np.zeros_like(periods)

    delta = sig_level / 2.0

    for i, scale in enumerate(periods):

        # Get the value from all simulations at this scale
        values = map(lambda x:x['global_ws'][i], all_results)

        p16, p50, p84 = np.percentile(values, [50.0 - delta, 50.0, 50.0 + delta])

        low_bound[i] = p16
        median[i] = p50
        hi_bound[i] = p84

    if plot:

        figure, sub = plt.subplots(1, 1)

        _ = sub.fill_between(periods, low_bound, hi_bound, alpha=0.5)
        _ = sub.plot(periods, median, lw=2, color='black')

        sub.set_xlabel("Time scale (s)")
        sub.set_ylabel("Power")

        sub.set_xscale("log")

        figure.tight_layout()

    else:

        figure = None

    return low_bound, median, hi_bound, figure


