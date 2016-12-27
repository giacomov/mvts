import astropy.io.fits as pyfits
import numpy as np
import warnings
import numexpr

from power_of_two_utils import next_power_of_2, is_power_of_2


class LightCurve(object):

    def __init__(self, tmin, tmax):

        self._tmin = tmin
        self._tmax = tmax

    def _fix_number_of_bins(self, target_bin_size):

        # Find the power of two that gives the closest (from below) binsize to the target binsize
        target_n_edges = int(np.ceil((self._tmax - self._tmin) / target_bin_size))
        target_n_bins = target_n_edges - 1

        # This finds the first power of 2 larger than target_n_bins
        n_bins = next_power_of_2(target_n_bins)
        n_edges = n_bins + 1

        edges, new_binsize = np.linspace(self._tmin, self._tmax, n_edges, retstep=True)

        print("Target binsize was %s, new bin size (to preserve power of 2 rule) is %s" % (target_bin_size,
                                                                                           new_binsize))

        return edges, new_binsize

    def get_light_curve(self, target_bin_size):

        raise NotImplementedError("You have to implement this in the derived class")


class LATLightCurve(LightCurve):

    def __init__(self, ft1_or_lle_file, ft2, emin=10.0, emax=100000.0, tmin=0, tmax=1e20, trigger_time=None):

        # Read data
        with pyfits.open(ft1_or_lle_file) as ft1_:

            data = ft1_['EVENTS'].data

            if trigger_time is None:

                trigger_time = ft1_['EVENTS'].header.get("TRIGTIME")

                if trigger_time is None:

                    trigger_time = ft1_[0].get("TRIGTIME")

        if trigger_time is None:

            raise IOError("Could not get TRIGTIME from event file, and you did not specify it during construction.")

        # Apply energy and time selection
        t = data.TIME - trigger_time
        e = data.ENERGY
        idx = (t >= tmin) & (t <= tmax) & (e >= emin) & (e <= emax)

        self._time = t[idx]

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

        self._ft2_entry_tstart = tstart[idx]
        self._ft2_entry_tstop = tstop[idx]
        self._livetime_fraction = livetime_fraction[idx]

        # Now sort all vectors
        idx = np.argsort(self._ft2_entry_tstart)

        self._ft2_entry_tstart = self._ft2_entry_tstart[idx]
        self._ft2_entry_tstop = self._ft2_entry_tstop[idx]
        self._livetime_fraction = self._livetime_fraction[idx]

        super(LATLightCurve, self).__init__(tmin, tmax)

    def _livetime_correction(self, bin_center):

        # Find the last tstart <= bin_center

        idx = np.searchsorted(self._ft2_entry_tstart, bin_center) - 1

        # Get the livetime for this bin
        this_livetime_fraction = self._livetime_fraction[idx]

        return this_livetime_fraction

    def get_light_curve(self, target_bin_size):

        # Histogram

        edges, new_binsize = self._fix_number_of_bins(target_bin_size)

        h, _ = np.histogram(self._time, edges)

        assert is_power_of_2(h.shape[0])

        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        # Livetime correction
        livetime_fractions = np.array(map(lambda x:self._livetime_correction(x), bin_centers))

        h_corrected = h / livetime_fractions

        assert np.all(h_corrected >= h)

        return h_corrected, bin_centers, edges, new_binsize


class GBMLightCurve(LightCurve):

    def __init__(self, tte_file, emin, emax, tmin=0, tmax=1e20, trigger_time=None):

        # Read data
        with pyfits.open(tte_file) as tte_:

            data = tte_['EVENTS'].data

            if trigger_time is None:

                trigger_time = tte_['EVENTS'].header.get("TRIGTIME")

                if trigger_time is None:
                    trigger_time = tte_[0].get("TRIGTIME")

            # Read EBOUNDS (needed by _energy_to_chan)
            self._ebounds = np.vstack([tte_['EBOUNDS'].data.field("E_MIN"),
                                       tte_['EBOUNDS'].data.field("E_MAX")]).T

            self._ebounds = self._ebounds.astype(float)

        if trigger_time is None:

            raise IOError("Could not get TRIGTIME from event file, and you did not specify it during construction.")

        # Apply time selection
        t = data.field("TIME") - trigger_time
        idx = (t >= tmin) & (t <= tmax)

        self._time = t[idx]
        self._pha = data.PHA[idx]
        self._last_channel = 128

        super(GBMLightCurve, self).__init__(tmin, tmax)

        # Compute and store deadtime
        self._compute_dead_time()

        # Now translate emin and emax to channels
        self._chan_min, self._chan_max = map(self._energy_to_channel, (emin, emax))

        print("Using events between channel %i and channel %i" % (self._chan_min, self._chan_max))

    def _compute_dead_time(self):
        """
        Computes an array of deadtime per event following the perscription of Meegan et al. (2009).

        """
        self._deadtime = np.zeros_like(self._time)
        overflow_mask = self._pha == 127  # specific to gbm! should work for CTTE

        # From Meegan et al. (2009)
        # Dead time for overflow (note, overflow sometimes changes)
        self._deadtime[overflow_mask] = 10.E-6  # s

        # Normal dead time
        self._deadtime[~overflow_mask] = 2.E-6  # s

    def _energy_to_channel(self, energy):

        '''Finds the channel containing the provided energy.
        NOTE: returns the channel index (starting at zero),
        not the channel number (likely starting from 1)'''

        # Get the index of the first ebounds upper bound larger than energy

        try:

            idx = next(idx for idx,
                               value in enumerate(self._ebounds[:, 1])
                       if value >= energy)

        except StopIteration:

            # No values above the given energy, return the last channel
            return self._ebounds[:, 1].shape[0]

        return idx

    def get_light_curve(self, target_bin_size, livetime_correction=False):

        edges, new_binsize = self._fix_number_of_bins(target_bin_size)

        if livetime_correction:
            h = np.zeros(edges.shape[0] - 1)

            # These are needed for numexpr
            _time = self._time
            _pha = self._pha
            _chan_min = self._chan_min
            _chan_max = self._chan_max

            for i, t1, t2 in zip(range(edges.shape[0] -1), edges[:-1], edges[1:]):

                # Count how many events are in this bin, within the requested energy range
                time_idx = numexpr.evaluate("(_time >= t1) & (_time < t2)")
                chan_idx = numexpr.evaluate("(_pha >= _chan_min) & (_pha <= _chan_max)")
                this_counts = np.sum(time_idx & chan_idx)

                # Now compute the deadtime, which depends on *all* events in this bin (not just the one between the
                # requested energies)
                deadtime = np.sum(self._deadtime[time_idx])
                livetime_fraction = (t2 - t1 - deadtime) / (t2 - t1)

                h[i] = this_counts / livetime_fraction

        else:

            # Choose events within the energy range
            _pha = self._pha
            _chan_min = self._chan_min
            _chan_max = self._chan_max
            chan_idx = numexpr.evaluate("(_pha >= _chan_min) & (_pha <= _chan_max)")

            h, _ = np.histogram(self._time[chan_idx], edges)

            h = np.array(h, float)

        assert is_power_of_2(h.shape[0])

        bin_centers = (edges[:-1] + edges[1:]) / 2.0

        return h, bin_centers, edges, new_binsize

