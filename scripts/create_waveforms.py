import gwmemory
import numpy as np
import matplotlib.pyplot as plt

q = 1.
m_tot = 60.
s1 = np.array([0., 0., 0.])
s2 = np.array([0., 0., 0.])
distance = 400.
l_max = 4
inc = np.pi / 2
phase = 0.

times = np.linspace(0, 16, 4096*16)
memory_generator_sur = gwmemory.waveforms.HybridSurrogate(q=q,
                                                          total_mass=m_tot,
                                                          spin_1=s1,
                                                          spin_2=s2,
                                                          l_max=l_max,
                                                          times=times,
                                                          distance=distance
                                                          )

memory_generator_imr = gwmemory.waveforms.Approximant(name='IMRPhenomD',
                                                      q=q,
                                                      MTot=m_tot,
                                                      S1=s1,
                                                      S2=s2,
                                                      distance=distance,
                                                      times=times)

memory_generator_mwm = gwmemory.waveforms.MWM(name='IMRPhenomD',
                                              q=q,
                                              MTot=m_tot,
                                              distance=distance,
                                              times=times)


phase_sur = 1.

h_oscillatory_sur_td, times_sur = memory_generator_sur.time_domain_oscillatory(times=times, inc=inc, phase=phase_sur)
h_oscillatory_imr_td = memory_generator_imr.time_domain_oscillatory(inc=inc, phase=phase)
# h_oscillatory_mwm_td = memory_generator_mwm.time_domain_oscillatory(inc=inc, phase=phase)

h_memory_sur_td, _ = memory_generator_sur.time_domain_memory(inc=inc, phase=phase_sur)
h_memory_imr_td, _ = memory_generator_imr.time_domain_memory(inc=inc, phase=phase)
h_memory_mwm_td, times_mwm = memory_generator_mwm.time_domain_memory(inc=inc, phase=phase)

plt.plot(times_sur, h_memory_sur_td['plus'], linestyle=':', label="SUR")
plt.plot(times, h_memory_imr_td['plus'], linestyle='--', label="IMR")
plt.plot(times_mwm, h_memory_mwm_td['plus'], linestyle='-.', label="MWM")
plt.legend()
plt.show()
plt.clf()


plt.plot(times_sur, h_oscillatory_sur_td['plus'], linestyle=':', label="SUR")
plt.plot(times, h_oscillatory_imr_td['plus'], linestyle='--', label="IMR")
# plt.plot(times_mwm, h_oscillatory_mwm_td['plus'], linestyle='-.', label="MWM")
plt.legend()
plt.show()
plt.clf()


### POSTER PLOT ###

# plt.plot(times, h_memory_td['plus'] + h_oscillatory_td['plus'], label='Memory + Oscillatory')
# plt.plot(times, h_memory_td['plus'], label='Memory')
# plt.plot([2.985, 2.985], [0, h_memory_td['plus'][-1]], color='red', linestyle='--', label='Permanent memory distortion')
# plt.legend()
# plt.axhline(y=0, color='black', linestyle=':')
# plt.xlim(2.85, 3)
# plt.xlabel('t[s]')
# plt.ylabel('$h$')
# plt.savefig(fname='total_h_plus_td_poster.pdf')
# plt.clf()

### END POSTER PLOT ###


# for mode in h_memory_fd:
#     total_memory_real_fd += h_memory_fd[mode].real
#     total_memory_imag_fd += h_memory_fd[mode].imag
#     plt.xlabel('f[Hz]')
#     plt.ylabel('$h_{memory}$')
#     plt.semilogx()
#     plt.plot(frequencies, h_memory_fd[mode].real)
#     plt.savefig(fname=str(mode) + 'real_fd')
#     plt.clf()
#     plt.xlabel('f[Hz]')
#     plt.semilogx()
#     plt.ylabel('$h_{memory}$')
#     plt.plot(frequencies, h_memory_fd[mode].imag)
#     plt.savefig(fname=str(mode) + 'imag_fd')
#     plt.clf()
#
# plt.plot(frequencies, total_memory_real_fd)
# plt.semilogx()
# plt.xlabel('f[Hz]')
# plt.ylabel('$h_{memory}$')
# plt.savefig(fname='total_real_fd')
# plt.clf()
#
# plt.plot(frequencies, total_memory_imag_fd)
# plt.semilogx()
# plt.xlabel('f')
# plt.ylabel('$h_{memory}$')
# plt.savefig(fname='total_imag_fd')
# plt.clf()
#
# ifo = bilby.gw.detector.get_empty_interferometer('L1')
# ifo.minimum_frequency = 0
# ifo.maximum_frequency = 1000000
# ifo.strain_data.frequency_array = frequencies
# ifo.strain_data.start_time = times[0]
# h_memory_response = ifo.get_detector_response(waveform_polarizations=dict(plus=total_memory_real_fd),
#                                               parameters=dict(ra=0,
#                                                               dec=0,
#                                                               geocent_time=0,
#                                                               psi=0))
#
# plt.plot(frequencies, h_memory_response)
# plt.semilogx()
# plt.xlabel('f[Hz]')
# plt.ylabel('$h_{memory}$')
# plt.savefig(fname='det_response_total_real_fd')
# plt.clf()
