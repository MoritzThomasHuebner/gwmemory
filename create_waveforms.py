import gwmemory
import numpy as np
import matplotlib.pyplot as plt
import bilby

h_memory_td, times = gwmemory.gwmemory.time_domain_memory(model='NRSur7dq2', q=1.5, MTot=60, S1=np.array([0, 0, 0]),
                                                          S2=np.array([0, 0, 0]), distance=400)

h_memory_fd, frequencies = gwmemory.gwmemory.frequency_domain_memory(model='NRSur7dq2', q=1.5, MTot=60,
                                                                     S1=np.array([0, 0, 0]),
                                                                     S2=np.array([0, 0, 0]), distance=400)

total_memory_real_td = np.zeros(len(times))
total_memory_imag_td = np.zeros(len(times))
for key in h_memory_td:
    total_memory_real_td += h_memory_td[key].real
    total_memory_imag_td += h_memory_td[key].imag
    plt.xlabel('t')
    plt.ylabel('h_{memory}')
    plt.plot(times, h_memory_td[key].real)
    plt.savefig(fname=str(key) + 'real_td')
    plt.clf()
    plt.xlabel('t')
    plt.ylabel('h_{memory}')
    plt.plot(times, h_memory_td[key].imag)
    plt.savefig(fname=str(key) + 'imag_td')
    plt.clf()

plt.plot(times, total_memory_real_td)
plt.xlabel('t')
plt.ylabel('h_{memory}')
plt.savefig(fname='total_real_td')
plt.clf()

plt.plot(times, total_memory_imag_td)
plt.xlabel('t')
plt.ylabel('h_{memory}')
plt.savefig(fname='total_imag_td')
plt.clf()

total_memory_real_fd = np.zeros(len(frequencies))
total_memory_imag_fd = np.zeros(len(frequencies))
for key in h_memory_fd:
    total_memory_real_fd += h_memory_fd[key].real
    total_memory_imag_fd += h_memory_fd[key].imag
    plt.xlabel('f')
    plt.ylabel('h_{memory}')
    plt.semilogx()
    plt.plot(frequencies, h_memory_fd[key].real)
    plt.savefig(fname=str(key) + 'real_fd')
    plt.clf()
    plt.xlabel('f')
    plt.semilogx()
    plt.ylabel('h_{memory}')
    plt.plot(frequencies, h_memory_fd[key].imag)
    plt.savefig(fname=str(key) + 'imag_fd')
    plt.clf()

plt.plot(frequencies, total_memory_real_fd)
plt.semilogx()
plt.xlabel('f')
plt.ylabel('h_{memory}')
plt.savefig(fname='total_real_fd')
plt.clf()

plt.plot(frequencies, total_memory_imag_fd)
plt.semilogx()
plt.xlabel('f')
plt.ylabel('h_{memory}')
plt.savefig(fname='total_imag_fd')
plt.clf()

ifo = bilby.gw.detector.get_empty_interferometer('L1')
ifo.minimum_frequency = 0
ifo.maximum_frequency = 1000000
ifo.strain_data.frequency_array = frequencies
ifo.strain_data.start_time = times[0]
h_memory_response = ifo.get_detector_response(waveform_polarizations=dict(plus=total_memory_real_fd),
                                              parameters=dict(ra=0,
                                                              dec=0,
                                                              geocent_time=0,
                                                              psi=0))

plt.plot(frequencies, h_memory_response)
plt.semilogx()
plt.xlabel('f')
plt.ylabel('h_{memory}')
plt.savefig(fname='det_response_total_real_fd')
plt.clf()
