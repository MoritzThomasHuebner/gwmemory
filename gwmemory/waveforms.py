from __future__ import print_function, division, absolute_import

from gwmemory import angles, qnms, utils, harmonics
import numpy as np
import lal
import lalsimulation as lalsim
import NRSur7dq2
from scipy.interpolate import interp1d
from .utils import cc, GG, Mpc, solar_mass
try:
    import gwsurrogate
    hybrid_surrogate = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
    nrsur7dq4_surrogate = gwsurrogate.LoadSurrogate('NRSur7dq4')
except ModuleNotFoundError as e:
    print(e)
    gwsurrogate = None
    hybrid_surrogate = None
    nrsur7dq4_surrogate = None


class MemoryGenerator(object):

    def __init__(self, name, h_lm=None, times=None, distance=None):
        self.name = name
        self.h_lm = h_lm
        self.times = times
        if self.h_lm is not None:
            try:
                self.zero_pad_h_lm()
            except KeyError:
                pass
        self.distance = distance
        self._modes = None

    @property
    def modes(self):
        if self._modes is None and self.h_lm is not None:
            self._modes = self.h_lm.keys()
        return self._modes

    @modes.setter
    def modes(self, modes):
        self._modes = modes

    @property
    def delta_t(self):
        return self.times[1] - self.times[0]

    @property
    def duration(self):
        return self.times[-1] - self.times[0]

    @property
    def sampling_frequency(self):
        return 1/(self.times[1] - self.times[0])

    @property
    def delta_f(self):
        return 1/self.duration

    def time_domain_memory(self, inc=None, phase=None, gamma_lmlm=None):
        """
        Calculate the spherical harmonic decomposition of the nonlinear memory from a dictionary of spherical mode time
        series

        Parameters
        ----------
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes will be returned.
        phase: float, optional
            Reference phase of the source, if None, the spherical harmonic modes will be returned.
            For CBCs this is the phase at coalesence.
        gamma_lmlm: dict
            Dictionary of arrays defining the angular dependence of the different memory modes, default=None
            if None the function will attempt to load them

        Return
        ------
        h_mem_lm: dict
            Time series of the spherical harmonic decomposed memory waveform.
        times: array_like
            Time series on which memory is evaluated.
        """
        if self.h_lm is None:
            _ = self.time_domain_oscillatory()
        lms = self.modes

        dhlm_dt = dict()
        for lm in lms:
            dhlm_dt[lm] = np.gradient(self.h_lm[lm], self.delta_t)

        dhlm_dt_sq = dict()
        for lm in lms:
            for lmp in lms:
                index = (lm, lmp)
                dhlm_dt_sq[index] = dhlm_dt[lm] * np.conjugate(dhlm_dt[lmp])

        if gamma_lmlm is None:
            gamma_lmlm = angles.load_gamma()

        # constant terms in SI units
        const = 1 / 4 / np.pi
        if self.distance is not None:
            const *= self.distance * utils.Mpc / utils.cc

        dh_mem_dt_lm = dict()
        for ii, ell in enumerate(gamma_lmlm['0'].l):
            if ell > 4:
                continue
            for delta_m in gamma_lmlm.keys():
                if abs(int(delta_m)) > ell:
                    continue
                dh_mem_dt_lm[(ell, int(delta_m))] = np.sum(
                    [dhlm_dt_sq[((l1, m1), (l2, m2))] * gamma_lmlm[delta_m]['{}{}{}{}'.format(l1, m1, l2, m2)][ii]
                     for (l1, m1), (l2, m2) in dhlm_dt_sq.keys() if m1 - m2 == int(delta_m)], axis=0)

        h_mem_lm = {lm: const * np.cumsum(dh_mem_dt_lm[lm]) * self.delta_t for lm in dh_mem_dt_lm}
        self.h_mem_lm = h_mem_lm
        if inc is None or phase is None:
            return h_mem_lm, self.times
        else:
            return combine_modes(h_mem_lm, inc, phase), self.times

    def time_domain_oscillatory(self, **kwargs):
        pass

    def set_time_array(self, times):
        """
        Change the time array on which the waveform is evaluated.
        
        Parameters
        ----------
        times: array_like
            New time array for waveform to be evaluated on.
        """
        for mode in self.modes:
            interpolated_mode = interp1d(self.times, self.h_lm)
            self.h_lm[mode] = interpolated_mode[times]
        self.times = times

    def zero_pad_h_lm(self):
        for lm in self.h_lm:
            self.h_lm[lm] = self.zero_pad_time_series(self.times, self.h_lm[lm])

    @staticmethod
    def zero_pad_time_series(times, mode):
        required_zeros = len(times) - len(mode)
        result = np.zeros(times.shape, dtype=np.complex128)
        if required_zeros > 0:
            result[:mode.shape[0]] = mode
            return result
        elif required_zeros < 0:
            return mode[-times.shape[0]:]
        else:
            return mode


class HybridSurrogate(MemoryGenerator):
    """
    Memory generator for a numerical relativity surrogate.
    Attributes
    ----------
    name: str
        Name of file to extract waveform from.
    modes: dict
        Spherical harmonic modes which we have knowledge of, default is ell<=4.
    h_lm: dict
        Spherical harmonic decomposed time-domain strain.
    times: array
        Array on which waveform is evaluated.
    q: float
        Binary mass ratio
    MTot: float, optional
        Total binary mass in solar units.
    distance: float, optional
        Distance to the binary in MPC.
    S1: array
        Spin vector of more massive black hole.
    S2: array
        Spin vector of less massive black hole.
    """

    def __init__(self, q, total_mass=None, spin_1=None,
                 spin_2=None, distance=None, l_max=4, modes=None, times=None,
                 minimum_frequency=10, reference_frequency=50., units='mks'):
        """
        Initialise Surrogate MemoryGenerator
        Parameters
        ----------
        name: str
            Name of the surrogate, default=NRSur7dq2.
        l_max: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        q: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in MPC.
        spin_1: array-like
            Spin vector of more massive black hole.
        spin_2: array-like
            Spin vector of less massive black hole.
        times: array-like
            Time array to evaluate the waveforms on, default is
            np.linspace(-900, 100, 10001).
        """
        self.sur = hybrid_surrogate

        self.q = q
        self.MTot = total_mass
        self.chi_1 = spin_1
        self.chi_2 = spin_2
        self.minimum_frequency = minimum_frequency
        self.distance = distance
        self.LMax = l_max
        self.modes = modes
        self.reference_frequency = reference_frequency
        self.units = units

        if total_mass is None:
            self.h_to_geo = 1
            self.t_to_geo = 1
        else:
            self.h_to_geo = self.distance * Mpc / self.MTot /\
                solar_mass / GG * cc ** 2
            self.t_to_geo = 1 / self.MTot / solar_mass / GG * cc ** 3

        self.h_lm = None
        self.times = times

        if times is not None and self.units == 'dimensionless':
            times *= self.t_to_geo

        h_lm, times = self.time_domain_oscillatory(modes=self.modes, times=times)

        MemoryGenerator.__init__(self, h_lm=h_lm, times=times, distance=distance, name='HybridSurrogate')

    def time_domain_oscillatory(self, times=None, modes=None, inc=None,
                                phase=None):
        """
        Get the mode decomposition of the surrogate waveform.
        Calculates a BBH waveform using the surrogate models of Field et al.
        (2014), Blackman et al. (2017)
        http://journals.aps.org/prx/references/10.1103/PhysRevX.4.031006,
        https://arxiv.org/abs/1705.07089
        See https://data.black-holes.org/surrogates/index.html for more
        information.
        Parameters
        ----------
        times: np.array, optional
            Time array on which to evaluate the waveform.
        modes: list, optional
            List of modes to try to generate.
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes
            will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic
            modes will be returned.
        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        MASS_TO_TIME = 4.925491025543576e-06
        dt = times[1] - times[0]
        duration = times[-1] - times[0] + dt
        times -= times[0]
        epsilon = 100 * MASS_TO_TIME * self.MTot
        t_NR = np.arange(-duration / 1.3 + epsilon, epsilon, dt)

        if self.h_lm is None:

            h_lm = self.sur([self.q, self.chi_1, self.chi_2], times=t_NR, f_low=0, M=self.MTot,
                            dist_mpc=self.distance, units='mks', f_ref=self.reference_frequency)

            del h_lm[(5, 5)]
            old_keys = [(ll, mm) for ll, mm in h_lm.keys()]
            for ll, mm in old_keys:
                if mm > 0:
                    h_lm[(ll, -mm)] = (- 1)**ll * np.conj(h_lm[(ll, mm)])

            available_modes = set(h_lm.keys())

            if modes is None:
                modes = available_modes

            if not set(modes).issubset(available_modes):
                print('Requested {} unavailable modes'.format(
                    ' '.join(set(modes).difference(available_modes))))
                modes = list(set(modes).union(available_modes))
                print('Using modes {}'.format(' '.join(modes)))

            h_lm = {(ell, m): h_lm[ell, m] for ell, m in modes}
            self.h_lm = h_lm
        else:
            h_lm = self.h_lm
            times = self.times
        t_NR -= t_NR[0]
        for mode in h_lm.keys():
            if len(times) != len(h_lm[mode]):
                h_lm[mode] = interp1d(t_NR, h_lm[mode], bounds_error=False, fill_value=0.0)(times)

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, phase), times

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        if q < 1:
            q = 1 / q
        if q > 8:
            raise ValueError('Surrogate waveform not valid for q>8.')
        self._q = q

    @property
    def chi_1(self):
        return self._chi_1

    @chi_1.setter
    def chi_1(self, spin_1):
        if spin_1 is None:
            self._chi_1 = 0.0
        elif len(np.atleast_1d(spin_1)) == 3:
            self._chi_1 = spin_1[2]
        else:
            self._chi_1 = spin_1

    @property
    def chi_2(self):
        return self._chi_2

    @chi_2.setter
    def chi_2(self, spin_2):
        if spin_2 is None:
            self._chi_2 = 0.0
        elif len(np.atleast_1d(spin_2)) == 3:
            self._chi_2 = spin_2[2]
        else:
            self._chi_2 = spin_2


class BaseSurrogate(MemoryGenerator):

    def __init__(self, q, name='', MTot=None, S1=None, S2=None, distance=None, LMax=4, max_q=2, times=None):

        MemoryGenerator.__init__(self, name=name, distance=distance)

        self.max_q = max_q
        self.q = q
        self.MTot = MTot
        self.S1 = S1
        self.S2 = S2
        self.LMax = LMax
        self.times = times

    @property
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        if q < 1:
            q = 1 / q
        if q > self.max_q:
            print(f'WARNING: Surrogate waveform not tested for q>{self.max_q}.')
        self.__q = q

    @property
    def S1(self):
        return self.__s1

    @S1.setter
    def S1(self, S1):
        if S1 is None:
            self.__s1 = np.array([0., 0., 0.])
        else:
            self.__s1 = np.array(S1)

    @property
    def S2(self):
        return self.__s2

    @S2.setter
    def S2(self, S2):
        if S2 is None:
            self.__s2 = np.array([0., 0., 0.])
        else:
            self.__s2 = np.array(S2)

    @property
    def h_to_geo(self):
        if self.MTot is None:
            return 1
        else:
            return self.distance * utils.Mpc / self.MTot / utils.solar_mass / utils.GG * utils.cc ** 2

    @property
    def t_to_geo(self):
        if self.MTot is None:
            return None
        else:
            return 1 / self.MTot / utils.solar_mass / utils.GG * utils.cc ** 3

    @property
    def geo_to_t(self):
        return 1 / self.t_to_geo

    @property
    def geometric_times(self):
        if self.times is not None:
            return self.times * self.t_to_geo
        else:
            return None


class Surrogate(BaseSurrogate):
    """
    Memory generator for a numerical relativity surrogate.

    Attributes
    ----------
    name: str
        Name of file to extract waveform from.
    modes: dict
        Spherical harmonic modes which we have knowledge of, default is ell<=4.
    h_lm: dict
        Spherical harmonic decomposed time-domain strain.
    times: array
        Array on which waveform is evaluated.
    q: float
        Binary mass ratio
    MTot: float, optional
        Total binary mass in solar units.
    distance: float, optional
        Distance to the binary in Mpc.
    S1: array
        Spin vector of more massive black hole.
    S2: array
        Spin vector of less massive black hole.
    """

    def __init__(self, q, name='', MTot=None, S1=None, S2=None, distance=None, LMax=4, modes=None, times=None):
        """
        Initialise Surrogate MemoryGenerator
        
        Parameters
        ----------
        name: str
            File name to load.
        LMax: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        q: float
            Binary mass ratio
        MTot: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in Mpc.
        S1: array_like
            Spin vector of more massive black hole.
        S2: array_like
            Spin vector of less massive black hole.
        times: array_like
            Time array to evaluate the waveforms on, default is np.linspace(-900, 100, 10001).
        """
        BaseSurrogate.__init__(self, q=q, name=name, MTot=MTot, S1=S1, S2=S2, distance=distance, LMax=LMax, times=times)
        self.sur = NRSur7dq2.NRSurrogate7dq2()
        self.h_lm, self.times = self.time_domain_oscillatory(modes=modes, times=self.geometric_times)

    def time_domain_oscillatory(self, times=None, modes=None, inc=None, phase=None):
        """
        Get the mode decomposition of the surrogate waveform.

        Calculates a BBH waveform using the surrogate models of Field et al. (2014), Blackman et al. (2017)
        http://journals.aps.org/prx/references/10.1103/PhysRevX.4.031006, https://arxiv.org/abs/1705.07089
        See https://data.black-holes.org/surrogates/index.html for more information.

        Parameters
        ----------
        times: np.array, optional
            Time array (In geometric units) on which to evaluate the waveform.
        modes: list, optional
            List of modes to try to generate.
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic modes will be returned.

        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        if self.h_lm is None:
            if times is None:
                times = self.default_geometric_times
            times = times * self.geo_to_t
            h_lm = self.sur(self.q, self.S1, self.S2, MTot=self.MTot, distance=self.distance, t=times, LMax=self.LMax)

            available_modes = set(h_lm.keys())

            if modes is None:
                modes = available_modes

            if not set(modes).issubset(available_modes):
                print('Requested {} unavailable modes'.format(' '.join(set(modes).difference(available_modes))))
                modes = list(set(modes).union(available_modes))
                print('Using modes {}'.format(' '.join(modes)))

            h_lm = {(ell, m): h_lm[ell, m] for ell, m in modes}

        else:
            h_lm = self.h_lm
            times = self.times

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, phase), times

    @property
    def times(self):
        return self.__times

    @times.setter
    def times(self, times):
        self.__times = times
        if times is None:
            return
        if max(self.geometric_times) > max(self.default_geometric_times):
            print("Warning: Time array exceeds the maximum allowed by NRSurrogate.\n Geometric time array max {} > "
                  "NRSurrogate max {}".format(max(self.geometric_times), max(self.default_geometric_times)))
        if min(self.geometric_times) > min(self.default_geometric_times):
            print("Warning: Time array exceeds the minimum allowed by NRSurrogate.\n Geometric time array min {} > "
                  "NRSurrogate min {}".format(min(self.geometric_times), min(self.default_geometric_times)))

    @property
    def default_geometric_times(self):
        return np.linspace(-900, 100, 10001)


class NRSur7dq4(BaseSurrogate):

    """
    Memory generator for a numerical relativity surrogate.
    Attributes
    ----------
    modes: dict
        Spherical harmonic modes which we have knowledge of, default is ell<=4.
    h_lm: dict
        Spherical harmonic decomposed time-domain strain.
    times: array
        Array on which waveform is evaluated.
    q: float
        Binary mass ratio
    MTot: float, optional
        Total binary mass in solar units.
    distance: float, optional
        Distance to the binary in MPC.
    S1: array
        Spin vector of more massive black hole.
    S2: array
        Spin vector of less massive black hole.
    """

    def __init__(self, q, total_mass=None, S1=None, S2=None, distance=None, l_max=4, modes=None, times=None,
                 minimum_frequency=20., reference_frequency=20., units='mks'):
        """
        Initialise Surrogate MemoryGenerator
        Parameters
        ----------
        name: str
            Name of the surrogate, default=NRSur7dq2.
        l_max: int
            Maximum ell value for oscillatory time series.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        q: float
            Binary mass ratio
        total_mass: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in MPC.
        spin_1: array-like
            Spin vector of more massive black hole.
        spin_2: array-like
            Spin vector of less massive black hole.
        times: array-like
            Time array to evaluate the waveforms on, default is
            np.linspace(-900, 100, 10001).
        """
        self.sur = nrsur7dq4_surrogate

        self.minimum_frequency = minimum_frequency
        self.reference_frequency = reference_frequency
        self.units = units
        self.l_max = l_max
        self.approximant = lalsim.GetApproximantFromString("NRSur7dq4")
        self.h_lm = None
        self.dt = times[1] - times[0]
        self.AVAILABLE_MODES = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (3, -3), (3, -2),
                                (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (4, -4), (4, -3),
                                (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
        super().__init__(q=q, name='NRSur7dq4', MTot=total_mass, S1=S1, S2=S2,
                         distance=distance, LMax=l_max, max_q=4, times=times)
        if modes is None:
            self.modes = self.AVAILABLE_MODES
        else:
            self.modes = modes
        self.h_lm, self.times = self.time_domain_oscillatory(modes=self.modes, times=times)

    def time_domain_oscillatory(self, modes=None, times=None, inc=None, phase=None):
        if times is None:
            times = self.times

        if modes is None:
            modes = self.modes

        times -= times[0]
        if self.h_lm is None:
            lal_params = lal.CreateDict()
            data = lalsim.SimInspiralChooseTDModes(
                0.0, self.dt, self.m1_SI, self.m2_SI, self.S1[0], self.S1[1], self.S1[2],
                self.S2[0], self.S2[1], self.S2[2], self.minimum_frequency,
                self.reference_frequency, self.distance_SI, lal_params, self.l_max, self.approximant)
            h_lm = {(l, m): lalsim.SphHarmTimeSeriesGetMode(data, l, m).data.data
                    for l, m in modes}
            t = np.arange(len(h_lm[list(h_lm.keys())[0]])) * self.delta_t
        else:
            h_lm = self.h_lm
            times = self.times
            t = self.times

        for mode in h_lm.keys():
            if len(times) != len(h_lm[mode]):
                h_lm[mode] = interp1d(t, h_lm[mode], bounds_error=False, fill_value=0.0)(times)

        if inc is None or phase is None:
            return h_lm, times
        else:
            return combine_modes(h_lm, inc, np.pi/2 - phase), times

    def time_domain_oscillatory_from_polarisations(self, inc, phase):

        lal_params = lal.CreateDict()

        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            self.m1_SI, self.m2_SI, self.S1[0], self.S1[1], self.S1[2], self.S2[0], self.S2[1], self.S2[2],
            self.distance_SI, inc, phase, 0.0, 0.0, 0.0, self.delta_t, self.minimum_frequency,
            self.reference_frequency, lal_params, self.approximant)
        hpc = dict(plus=hp.data.data, cross=hc.data.data)

        required_zeros = len(self.times) - len(hpc['plus'])
        if required_zeros == 0:
            return hpc
        elif required_zeros > 0:
            for mode in hpc:
                result = np.zeros(self.times.shape, dtype=np.complex128)
                result[:hpc[mode].shape[0]] = hpc[mode]
                hpc[mode] = result
        elif required_zeros < 0:
            for mode in hpc:
                hpc[mode] = hpc[mode][-self.times.shape[0]:]
        return hpc

    @property
    def m1(self):
        return self.MTot / (1 + self.q)

    @property
    def m2(self):
        return self.m1 * self.q

    @property
    def m1_SI(self):
        return self.m1 * utils.solar_mass

    @property
    def m2_SI(self):
        return self.m2 * utils.solar_mass

    @property
    def distance_SI(self):
        return self.distance * utils.Mpc


class SXSNumericalRelativity(MemoryGenerator):
    """
    Memory generator for a numerical relativity waveform.

    Attributes
    ----------
    name: str
        Name of file to extract waveform from.
    modes: dict
        Spherical harmonic modes which we have knowledge of, default is ell<=4.
    h_lm: dict
        Spherical harmonic decomposed time-domain strain.
    times: array
        Array on which waveform is evaluated.
    MTot: float, optional
        Total binary mass in solar units.
    distance: float, optional
        Distance to the binary in Mpc.
    """

    def __init__(self, name, modes=None, extraction='OutermostExtraction.dir', MTot=None, distance=None, times=None):
        """
        Initialise SXSNumericalRelativity MemoryGenerator
        
        Parameters
        ----------
        name: str
            File name to load.
        modes: dict, optional
            Modes to load in, default is all ell<=4.
        extraction: str
            Extraction method, this specifies the outer object to use in the h5 file.
        MTot: float
            Lab-frame total mass of the binary in solar masses.
        distance: float
            Luminosity distance to the binary in Mpc.
        times: array_like
            Time array to evaluate the waveforms on, default is time array in h5 file.
        """
        self.name = name
        self.h_lm, self.times = utils.load_sxs_waveform(name, modes=modes, extraction=extraction)

        self.MTot = MTot
        self.distance = distance

        if MTot is None or distance is None:
            self.h_to_geo = 1
            self.t_to_geo = 1
        else:
            self.h_to_geo = self.distance * utils.Mpc / self.MTot / utils.solar_mass / utils.GG * utils.cc ** 2
            self.t_to_geo = 1 / self.MTot / utils.solar_mass / utils.GG * utils.cc ** 3

            for mode in self.h_lm:
                self.h_lm /= self.h_to_geo
            self.times / self.t_to_geo
            # Rezero time array to the merger time
            self.times -= self.times[np.argmax(abs(self.h_lm[(2, 2)]))]

        if times is not None:
            self.set_time_array(times)

        MemoryGenerator.__init__(self, name=name, h_lm=self.h_lm, times=times, distance=self.distance)

    def time_domain_oscillatory(self, times=None, modes=None, inc=None, phase=None):
        """
        Get the mode decomposition of the numerical relativity waveform.

        Parameters
        ----------
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic modes will be returned.

        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        if inc is None or phase is None:
            return self.h_lm, times
        else:
            return combine_modes(self.h_lm, inc, phase), times


class Approximant(MemoryGenerator):

    def __init__(self, name, q, MTot=60, S1=np.array([0, 0, 0]), S2=np.array([0, 0, 0]), distance=400, times=None):
        """
        Initialise Surrogate MemoryGenerator
        
        Parameters
        ----------
        name: str
            File name to load.
        q: float
            Binary mass ratio
        MTot: float, optional
            Total binary mass in solar units.
        distance: float, optional
            Distance to the binary in Mpc.
        S1: array_like
            Spin vector of more massive black hole.
        S2: array_like
            Spin vector of less massive black hole.
        times: array_like
            Time array to evaluate the waveforms on, default is time array from lalsimulation.
            FIXME
        """
        self.q = q
        self.MTot = MTot
        self.__S1 = S1
        self.__S2 = S2
        self._check_prececssion()

        MemoryGenerator.__init__(self, name=name, times=times, distance=distance)

    @property
    def available_modes(self):
        return [(2, 2), (2, -2)]

    @property
    def m1(self):
        return self.MTot / (1 + self.q)

    @property
    def m2(self):
        return self.m1 * self.q

    @property
    def m1_SI(self):
        return self.m1 * utils.solar_mass

    @property
    def m2_SI(self):
        return self.m2 * utils.solar_mass

    @property
    def distance_SI(self):
        return self.distance * utils.Mpc

    @property
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        if q > 1:
            q = 1 / q
        self.__q = q

    @property
    def h_to_geo(self):
        return self.distance_SI / (self.m1_SI + self.m2_SI) / utils.GG * utils.cc ** 2

    @property
    def t_to_geo(self):
        return 1 / (self.m1_SI + self.m2_SI) / utils.GG * utils.cc ** 3

    @property
    def delta_t(self):
        return self.times[1] - self.times[0]

    @property
    def S1(self):
        return self.__S1

    @S1.setter
    def S1(self, S1):
        if S1 is None:
            self.__S1 = np.array([0., 0., 0.])
        else:
            self.__S1 = np.array(S1)
        self._check_prececssion()

    @property
    def S2(self):
        return self.__S2

    @S2.setter
    def S2(self, S2):
        if S2 is None:
            self.__S2 = np.array([0., 0., 0.])
        else:
            self.__S2 = np.array(S2)
        self._check_prececssion()

    @property
    def approximant(self):
        return lalsim.GetApproximantFromString(self.name)

    def _check_prececssion(self):
        if abs(self.__S1[0]) > 0 or abs(self.__S1[1]) > 0 or abs(self.__S2[0]) > 0 or abs(self.__S2[1]) > 0:
            print('WARNING: Approximant decomposition works only for non-precessing waveforms.')
            print('Setting spins to be aligned')
            self.__S1[0], self.__S1[1] = 0., 0.
            self.__S2[0], self.__S2[1] = 0., 0.
            print('New spins are: S1 = {}, S2 = {}'.format(self.__S1, self.__S2))
        else:
            self.__S1 = list(self.__S1)
            self.__S2 = list(self.__S2)

    def time_domain_oscillatory(self, modes=None, inc=None, phase=None):
        """
        Get the mode decomposition of the waveform approximant.

        Since the waveforms we consider only contain content about the ell=|m|=2 modes.
        We can therefore evaluate the waveform for a face-on system, where only the (2, 2) mode
        is non-zero.

        Parameters
        ----------
        modes: list, optional
            List of modes to try to generate.
        inc: float, optional
            Inclination of the source, if None, the spherical harmonic modes will be returned.
        phase: float, optional
            Phase at coalescence of the source, if None, the spherical harmonic modes will be returned.

        Returns
        -------
        h_lm: dict
            Spin-weighted spherical harmonic decomposed waveform.
        times: np.array
            Times on which waveform is evaluated.
        """
        if self.h_lm is None:
            if modes is None:
                modes = self.available_modes
            else:
                modes = modes

            if not set(modes).issubset(self.available_modes):
                print('Requested {} unavailable modes'.format(' '.join(set(modes).difference(self.available_modes))))
                modes = list(set(modes).union(self.available_modes))
                print('Using modes {}'.format(' '.join(modes)))

            fmin = 20.
            fRef = 20
            theta = 0.0
            phi = 0.0
            longAscNodes = 0.0
            eccentricity = 0.0
            meanPerAno = 0.0
            WFdict = lal.CreateDict()

            hplus, hcross = lalsim.SimInspiralChooseTDWaveform(
                self.m1_SI, self.m2_SI, self.S1[0], self.S1[1], self.S1[2], self.S2[0], self.S2[1], self.S2[2],
                self.distance_SI, theta, phi, longAscNodes, eccentricity, meanPerAno, self.delta_t, fmin, fRef,
                WFdict, self.approximant)

            h = hplus.data.data - 1j * hcross.data.data
            h_22 = h / harmonics.sYlm(-2, 2, 2, theta, phi)

            self.h_lm = {(2, 2): h_22, (2, -2): np.conjugate(h_22)}
            self.zero_pad_h_lm()

        if inc is None or phase is None:
            return self.h_lm
        else:
            return combine_modes(h_lm=self.h_lm, inc=inc, phase=phase)


class PhenomXHM(Approximant):

    def __init__(self, q, MTot=60, S1=np.array([0, 0, 0]), S2=np.array([0, 0, 0]), distance=400, times=None):
        name = "IMRPhenomXHM"
        super().__init__(name, q, MTot, S1, S2, distance, times)

    @property
    def available_modes(self):
        return [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (3, 2), (3, -2), (4, 4), (4, -4)]


    def time_domain_oscillatory(self, modes=None, inc=None, phase=None):
        if self.h_lm is None:
            if modes is None:
                modes = self.available_modes

            self.h_lm = dict()
            for mode in modes:
                h_lm, times = self.single_mode_from_choose_td(l=mode[0], m=mode[1], mbandthreshold=0)
                self.h_lm[mode] = h_lm
            self.zero_pad_h_lm()

        if inc is None or phase is None:
            return self.h_lm
        else:
            return combine_modes(h_lm=self.h_lm, inc=inc, phase=phase)

    def time_domain_oscillatory_from_polarisations(self, inc, phase):
        hpc, _ = self.polarisations(mbandthreshold=0, inc=inc, phase=phase)
        for mode in hpc:
            hpc[mode] = self.zero_pad_time_series(times=self.times, mode=hpc[mode])
        return hpc

    def single_mode_from_choose_td(self, l, m, mbandthreshold):
        inc = 0.4
        phi = np.pi / 2
        lalparams = self._get_single_mode_lalparams(l, m, mbandthreshold)

        hpc, times = self.get_polarisations(inc=inc, phase=phi, lalparams=lalparams)
        hlm = (hpc['plus'] - 1j * hpc['cross']) / lal.SpinWeightedSphericalHarmonic(inc, np.pi - phi, -2, l, m)

        return hlm, times

    @staticmethod
    def _get_single_mode_lalparams(l, m, mbandthreshold):
        lalparams = lal.CreateDict()
        ModeArray = lalsim.SimInspiralCreateModeArray()
        lalsim.SimInspiralModeArrayActivateMode(ModeArray, l, m)
        lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
        if mbandthreshold != None:
            lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalparams, mbandthreshold)
        return lalparams

    def polarisations(self, mbandthreshold, inc, phase):
        lalparams = lal.CreateDict()
        if mbandthreshold != None:
            lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalparams, mbandthreshold)
        hpc, times = self.get_polarisations(inc=inc, phase=phase, lalparams=lalparams)

        return hpc, times

    def get_polarisations(self, inc, phase, lalparams):
        f_min = 20.
        f_ref = 20.
        longAscNodes = 0.0
        eccentricity = 0.0
        meanPerAno = 0.0

        hp, hc = lalsim.SimInspiralChooseTDWaveform(
            self.m1_SI, self.m2_SI, self.S1[0], self.S1[1], self.S1[2], self.S2[0], self.S2[1], self.S2[2],
            self.distance_SI, inc, phase, longAscNodes, eccentricity, meanPerAno, self.delta_t, f_min, f_ref,
            lalparams, lalsim.IMRPhenomXHM)

        shift = hp.epoch.gpsSeconds + hp.epoch.gpsNanoSeconds / 1e9
        times = np.arange(len(hp.data.data)) * self.delta_t + shift
        hpc = dict(plus=hp.data.data, cross=hc.data.data)
        return hpc, times


class MWM(MemoryGenerator):

    def __init__(self, q, MTot=60, distance=400, name='MWM', times=None):
        MemoryGenerator.__init__(self, name=name, h_lm=dict(), times=times, distance=distance)
        self.name = name
        if q > 1:
            q = 1 / q
        self.q = q
        self.MTot = MTot
        self.m1 = self.MTot / (1 + self.q)
        self.m2 = self.m1 * self.q

        self.h_to_geo = self.distance * utils.Mpc / (self.m1 + self.m2) / utils.solar_mass \
                        / utils.GG * utils.cc ** 2
        self.t_to_geo = 1 / (self.m1 + self.m2) / utils.solar_mass / utils.GG * utils.cc ** 3

        if times is None:
            times = np.linspace(-900, 100, 10001) / self.t_to_geo
        self.times = times

    def time_domain_memory(self, inc, phase, times=None, rm=3):
        """
        Calculates the plus and cross polarisations for the
        minimal waveform model memory waveform:
        eqns (5) and (9) from Favata (2009. ApJL 159)

        TODO: Implement spherical harmonic decomposition?

        Parameters
        ----------
        inc: float
            Binary inclination angle
        phase: float
            Binary phase at coalscence
        times: array, optional
            Time array on which the memory is calculated
        rm: float, optional
            Radius at which the matching occurs in solar masses

        Returns
        -------
        h_mem: dict
            Plus and cross polarisations of the memory waveform
        times: np.array
            Time array on which the memory is calculated

        Paul Lasky
        """
        if times is None:
            times = self.times

        time_geo = utils.time_s_to_geo(times)  # units: metres

        m1_geo = utils.m_sol_to_geo(self.m1)  # units: metres
        m2_geo = utils.m_sol_to_geo(self.m2)  # units: metres

        dist_geo = utils.dist_Mpc_to_geo(self.distance)  # units: metres

        # total mass
        MM = m1_geo + m2_geo

        # symmetric mass ratio
        eta = utils.m12_to_symratio(m1_geo, m2_geo)

        # this is the orbital separation at the matching radius -- see Favata (2009) before eqn (8).
        # the default value for this is given as rm = 3 MM.
        rm *= MM

        # calculate dimensionless mass and spin of the final black hole
        # from the Buonanno et al. (2007) fits
        Mf_geo, jj = qnms.final_mass_spin(m1_geo, m2_geo)

        # calculate the QNM frequencies and damping times
        # from the fits in Table VIII of Berti et al. (2006)
        omega220, tau220 = qnms.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=0)
        omega221, tau221 = qnms.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=1)
        omega222, tau222 = qnms.freq_damping(Mf_geo, jj, ell=2, mm=2, nn=2)

        sigma220 = 1j * omega220 + 1 / tau220
        sigma221 = 1j * omega221 + 1 / tau221
        sigma222 = 1j * omega222 + 1 / tau222

        # set the matching time to be at t = 0
        tm = 0
        TT = time_geo - tm

        # some quantity defined after equation (7) of Favata
        trr = 5 * MM * rm ** 4 / (256 * eta * MM ** 4)

        # calculate the A_{ell m n} matching coefficients.  Note that I've solved
        # a matrix equation that solved for the three coefficients fron three equations
        xi = 2 * np.sqrt(2 * np.pi / 5) * eta * MM * rm ** 2
        chi = -2 * 1j * np.sqrt(MM / rm ** 3)

        A220 = xi * (sigma221 * sigma222 * chi ** 2 + sigma221 * chi ** 3 + sigma222 * chi ** 3 + chi ** 4) \
               / ((sigma220 - sigma221) * (sigma220 - sigma222))
        A221 = xi * (sigma220 * sigma222 * chi ** 2 + sigma220 * chi ** 3 + sigma222 * chi ** 3 + chi ** 4) \
               / ((sigma221 - sigma220) * (sigma221 - sigma222))
        A222 = xi * (sigma220 * sigma221 * chi ** 2 + sigma220 * chi ** 3 + sigma221 * chi ** 3 + chi ** 4) \
               / ((sigma221 - sigma222) * (sigma220 - sigma222))

        # Calculate the coefficients in the summed term of equation (9) from Favata (2009)
        # this is a double sum, with each variable going from n = 0 to 2; therefore 9 terms
        coeffSum00 = sigma220 * np.conj(sigma220) * A220 * np.conj(A220) / (sigma220 + np.conj(sigma220))
        coeffSum01 = sigma220 * np.conj(sigma221) * A220 * np.conj(A221) / (sigma220 + np.conj(sigma221))
        coeffSum02 = sigma220 * np.conj(sigma222) * A220 * np.conj(A222) / (sigma220 + np.conj(sigma222))

        coeffSum10 = sigma221 * np.conj(sigma220) * A221 * np.conj(A220) / (sigma221 + np.conj(sigma220))
        coeffSum11 = sigma221 * np.conj(sigma221) * A221 * np.conj(A221) / (sigma221 + np.conj(sigma221))
        coeffSum12 = sigma221 * np.conj(sigma222) * A221 * np.conj(A222) / (sigma221 + np.conj(sigma222))

        coeffSum20 = sigma222 * np.conj(sigma220) * A222 * np.conj(A220) / (sigma222 + np.conj(sigma220))
        coeffSum21 = sigma222 * np.conj(sigma221) * A222 * np.conj(A221) / (sigma222 + np.conj(sigma221))
        coeffSum22 = sigma222 * np.conj(sigma222) * A222 * np.conj(A222) / (sigma222 + np.conj(sigma222))

        # radial separation
        rr = rm * (1 - TT / trr) ** (1 / 4)

        # set up strain
        h_MWM = np.zeros(len(TT))

        # calculate strain for TT < 0.
        h_MWM[TT <= 0] = 8 * np.pi * MM / rr[TT <= 0]

        # calculate strain for TT > 0.
        term00 = coeffSum00 * (1 - np.exp(-TT[TT > 0] * (sigma220 + np.conj(sigma220))))
        term01 = coeffSum01 * (1 - np.exp(-TT[TT > 0] * (sigma220 + np.conj(sigma221))))
        term02 = coeffSum02 * (1 - np.exp(-TT[TT > 0] * (sigma220 + np.conj(sigma222))))

        term10 = coeffSum10 * (1 - np.exp(-TT[TT > 0] * (sigma221 + np.conj(sigma220))))
        term11 = coeffSum11 * (1 - np.exp(-TT[TT > 0] * (sigma221 + np.conj(sigma221))))
        term12 = coeffSum12 * (1 - np.exp(-TT[TT > 0] * (sigma221 + np.conj(sigma222))))

        term20 = coeffSum20 * (1 - np.exp(-TT[TT > 0] * (sigma222 + np.conj(sigma220))))
        term21 = coeffSum21 * (1 - np.exp(-TT[TT > 0] * (sigma222 + np.conj(sigma221))))
        term22 = coeffSum22 * (1 - np.exp(-TT[TT > 0] * (sigma222 + np.conj(sigma222))))

        sum_terms = np.real(term00 + term01 + term02 +
                            term10 + term11 + term12 +
                            term20 + term21 + term22)

        h_MWM[TT > 0] = 8 * np.pi * MM / rm + sum_terms / (eta * MM)

        # calculate the plus polarisation of GWs: eqn. (5) from Favata (2009)
        sT = np.sin(inc)
        cT = np.cos(inc)

        h_plus_coeff = 0.77 * eta * MM / (384 * np.pi) * sT ** 2 * (17 + cT ** 2) / dist_geo
        h_mem = dict(plus=h_plus_coeff * h_MWM, cross=np.zeros_like(h_MWM))

        return h_mem, times


def combine_modes(h_lm, inc, phase):
    """Calculate the plus and cross polarisations of the waveform from the spherical harmonic decomposition."""
    # total = sum([h_lm[(l, m)] * harmonics.sYlm(-2, l, m, inc, phase) for l, m in h_lm])
    total = sum([h_lm[(l, m)] * lal.SpinWeightedSphericalHarmonic(inc, phase, -2, l, m) for l, m in h_lm])
    h_plus_cross = dict(plus=total.real, cross=-total.imag)
    return h_plus_cross


combine_modes_vectorized = np.vectorize(pyfunc=combine_modes, excluded=['h_lm', 'inc'], otypes=[dict])
