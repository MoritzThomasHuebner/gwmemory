import gwmemory
import numpy as np
import matplotlib.pyplot as plt

parameters = dict(
    mass_1=35.0,
    mass_2=30.0,
    eccentricity=0.0,
    luminosity_distance=440.0,
    theta_jn=1.5,
    psi=0.1,
    phase=1.2,
    geocent_time=0.0,
    ra=0.3,
    dec=0.7,
    chi_1=0.0,
    chi_2=0.0
)

memory_generator = gwmemory.waveforms.Eccentric(q=0.8, MTot=60, e=0.0)
memory_generator_2 = gwmemory.waveforms.Eccentric(q=0.8, MTot=60, e=0.3)


# h_oscillatory_td, times = memory_generator.time_domain_oscillatory(inc=parameters['theta_jn'], phase=parameters['phase'])
# h_oscillatory_td, times = memory_generator_2.time_domain_oscillatory(inc=parameters['theta_jn'], phase=parameters['phase'])

h_memory_td, times = memory_generator.time_domain_memory(inc=parameters['theta_jn'], phase=parameters['phase'])
h_memory_td_2, times_2 = memory_generator_2.time_domain_memory(inc=parameters['theta_jn'], phase=parameters['phase'])
plt.plot(times, h_memory_td['plus'], label="e = 0")
plt.plot(times_2, h_memory_td_2['plus'], label="e = 0.3")
plt.xlim(-0.4, 0.1)
plt.legend()
# plt.show()
plt.savefig('test.pdf')
plt.clf()
