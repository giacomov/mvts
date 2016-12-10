import astropy.io.fits as pyfits
import numpy as np
import warnings

from power_of_two_utils import next_power_of_2, is_power_of_2


class LATLightCurve(object):

    def __init__(self, ft1_or_lle_file, ft2, emin=10.0, emax=100000.0, tmin=0, tmax=1e20, trigger_time=None):

        # Read data
        with pyfits.open(ft1_or_lle_file) as ft1_:

            data = ft1_['EVENTS'].data

            if trigger_time is None:

                trigger_time = ft1_['EVENTS'].header.get("TRIGTIME")

                if trigger_time is None:

                    trigger_time = ft1_[0].get("TRIGTIME")

        # Apply energy and time selection
        t = data.TIME - trigger_time
        e = data.ENERGY
        idx = (t >= tmin) & (t <= tmax) & (e >= emin) & (e <= emax)

        if trigger_time is None:
            raise IOError("Could not get TRIGTIME from event file, and you did not specify it during construction.")

        self._time = t[idx]
        self._tmin = tmin
        self._tmax = tmax

        # Read FT2 data
        with pyfits.open(ft2) as ft2_:

            tstart = ft2_['SC_DATA'].data.field("START") - trigger_time
            tstop = ft2_['SC_DATA'].data.field("STOP") - trigger_time
            livetime = ft2_['SC_DATA'].data.field("LIVETIME")

        ft2_bin_size = 1.0 # seconds

        if not np.all(livetime <= 1.0):

            warnings.warn("You are using a 30s FT2 file. You should use a 1s Ft2 file otherwise the livetime "
                          "correction will not be accurate!")

            ft2_bin_size = 30.0 # s

        livetime_fraction = livetime / (tstop - tstart)

        # Keep only the needed entries (plus a padding)
        idx = (tstart >= tmin - 10 * ft2_bin_size) & (tstop <= tmax + 10 * ft2_bin_size)

        self._tstart = tstart[idx]
        self._tstop = tstop[idx]
        self._livetime_fraction = livetime_fraction[idx]

        # Now sort all vectors
        idx = np.argsort(self._tstart)

        self._tstart = self._tstart[idx]
        self._tstop = self._tstop[idx]
        self._livetime_fraction = self._livetime_fraction[idx]

    def _livetime_correction(self, bin_center):

        # Find the last tstart <= bin_center

        idx = np.searchsorted(self._tstart, bin_center) - 1

        # Get the livetime for this bin
        this_livetime_fraction = self._livetime_fraction[idx]

        return this_livetime_fraction

    def get_light_curve(self, target_binsize):

        # Histogram

        # Find the power of two that gives the closest (from below) binsize to the target binsize
        target_n_edges = int(np.ceil((self._tmax - self._tmin) / target_binsize))
        target_n_bins = target_n_edges - 1

        # This finds the first power of 2 larger than target_n_bins
        n_bins = next_power_of_2(target_n_bins)
        n_edges = n_bins + 1

        edges, new_binsize = np.linspace(self._tmin, self._tmax, n_edges, retstep=True)

        print("Target binsize was %s, new bin size (to preserve power of 2 rule) is %s" % (target_binsize, new_binsize))

        h, _ = np.histogram(self._time, edges)

        assert is_power_of_2(h.shape[0])

        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        # Livetime correction
        livetime_fractions = np.array(map(lambda x:self._livetime_correction(x), bin_centers))

        h_corrected = h / livetime_fractions

        assert np.all(h_corrected >= h)

        return h_corrected, bin_centers, edges, new_binsize








