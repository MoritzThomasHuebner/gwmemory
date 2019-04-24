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
memory_generator = gwmemory.waveforms.HybridSurrogate(q=q,
                                                      total_mass=m_tot,
                                                      minimum_frequency=10,
                                                      spin_1=s1,
                                                      spin_2=s2,
                                                      l_max=l_max,
                                                      times=times,
                                                      distance=distance
                                                      )

# memory_generator = gwmemory.waveforms.Approximant(name='IMRPhenomD',
#                                                   q=q,
#                                                   MTot=m_tot,
#                                                   S1=s1,
#                                                   S2=s2,
#                                                   distance=distance,
#                                                   times=times)

# h_oscillatory_td, times = memory_generator.time_domain_oscillatory(times=times, inc=inc, phase=phase)
h_oscillatory_td, times = memory_generator.time_domain_oscillatory(inc=inc, phase=phase)
h_memory_td, times = memory_generator.time_domain_memory(inc=inc, phase=phase)
print(np.max(h_oscillatory_td['plus']))
print(np.max(h_oscillatory_td['cross']))
print(np.max(h_memory_td['plus']))
print(np.max(h_memory_td['cross']))
# h_memory_fd, frequencies = gwmemory.gwmemory.frequency_domain_memory(model='NRSur7dq2', q=2, MTot=60,
#                                                                      S1=np.array([0, 0, 0]),
#                                                                      S2=np.array([0, 0, 0]), distance=400)

total_memory_real_td = np.zeros(len(times))
total_oscillatory_real_td = np.zeros(len(times))
total_memory_imag_td = np.zeros(len(times))
total_oscillatory_imag_td = np.zeros(len(times))
# total_memory_real_fd = np.zeros(len(frequencies))
# total_memory_imag_fd = np.zeros(len(frequencies))

for mode in h_memory_td:
    total_memory_real_td += h_memory_td[mode].real
    total_memory_imag_td += h_memory_td[mode].imag
    plt.xlabel('t[s]')
    plt.ylabel('$h_{memory}$')
    plt.plot(times, h_memory_td[mode].real)
    plt.savefig(fname=str(mode) + 'real_td')
    plt.clf()
    plt.xlabel('t[s]')
    plt.ylabel('$h_{memory}$')
    plt.plot(times, h_memory_td[mode].imag)
    plt.savefig(fname=str(mode) + 'imag_td')
    plt.clf()

for mode in h_oscillatory_td:
    plt.xlabel('t[s]')
    plt.ylabel('$h_{oscillatory}$')
    plt.plot(times, h_oscillatory_td[mode])
    plt.savefig(fname='total_h_oscillatory_' + mode + '_td')
    plt.clf()
    plt.plot(times, h_memory_td[mode])
    plt.xlabel('t[s]')
    plt.ylabel('$h_{memory}$')
    plt.savefig(fname='total_h_memory_' + mode + '_td')
    plt.clf()
    plt.plot(times, h_memory_td[mode] + h_oscillatory_td[mode])
    plt.xlabel('t[s]')
    plt.ylabel('$h$')
    plt.savefig(fname='total_h_' + mode + '_td')
    plt.clf()

### POSTER PLOT ###

plt.plot(times, h_memory_td['plus'] + h_oscillatory_td['plus'], label='Memory + Oscillatory')
plt.plot(times, h_memory_td['plus'], label='Memory')
plt.plot([2.985, 2.985], [0, h_memory_td['plus'][-1]], color='red', linestyle='--', label='Permanent memory distortion')
plt.legend()
plt.axhline(y=0, color='black', linestyle=':')
plt.xlim(2.85, 3)
plt.xlabel('t[s]')
plt.ylabel('$h$')
plt.savefig(fname='total_h_plus_td_poster.pdf')
plt.clf()

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
