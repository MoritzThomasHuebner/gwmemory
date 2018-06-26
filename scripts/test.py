from __future__ import division
import numpy as np
import tupak
import matplotlib.pyplot as plt
import waveform_functions

mass_ratio = 2
name = 'memester'
total_mass = 60
S1 = np.array([0.8, 0, 0])
S2 = np.array([0, 0.8, 0])
s11 = S1[0]
s12 = S1[1]
s13 = S1[2]
s21 = S2[0]
s22 = S2[1]
s23 = S2[2]
LMax = 3
luminosity_distance = 2000.
inc = np.pi / 2
pol = 0
ra = 1.375
dec = -1.2108
psi = 2.659
geocent_time = 1126259642.413

start_time = -0.5
end_time = 0.029

outdir = 'outdir'
label = 'test_2'

tupak.core.utils.setup_logger(outdir=outdir, label=label)
time_array = np.linspace(start_time, end_time, 1000)
time_duration = time_array[-1] - time_array[0]
sampling_frequency = tupak.utils.get_sampling_frequency(time_array)
frequency_array = tupak.utils.create_frequency_series(sampling_frequency=sampling_frequency,
                                                      duration=time_duration)

np.random.seed(88170235)

injection_parameters = dict(total_mass=total_mass, mass_ratio=mass_ratio, s11=s11, s12=s12, s13=s13, s21=s21,
                            s22=s22, s23=s23, luminosity_distance=luminosity_distance, inc=inc, pol=pol,
                            psi=psi, geocent_time=geocent_time, ra=ra, dec=dec, LMax=LMax)

waveform_generator = tupak.WaveformGenerator(time_duration=time_duration,
                                             sampling_frequency=sampling_frequency,
                                             time_domain_source_model=waveform_functions.
                                             time_domain_nr_sur_waveform_without_memory,
                                             parameters=injection_parameters,
                                             waveform_arguments=dict(LMax=LMax))
waveform_generator.time_array = time_array.copy()
waveform_generator.frequency_array = frequency_array
hf_signal = waveform_generator.frequency_domain_strain()
test_strain_signal = waveform_generator.time_domain_strain()
plt.plot(time_array, test_strain_signal['plus'])
plt.show()
IFOs = [tupak.gw.detector.get_interferometer_with_fake_noise_and_injection(
    name, injection_polarizations=hf_signal, injection_parameters=injection_parameters, time_duration=time_duration,
    sampling_frequency=sampling_frequency, start_time=start_time, outdir=outdir) for name in ['H1', 'L1']]

priors = dict()
for key in ['total_mass', 'mass_ratio', 's11', 's12', 's13', 's21', 's22', 's23', 'luminosity_distance',
            'inc', 'pol', 'ra', 'dec', 'geocent_time', 'psi']:
    priors[key] = injection_parameters[key]
priors['total_mass'] = tupak.prior.Uniform(minimum=50, maximum=70)
priors['luminosity_distance'] = tupak.prior.Uniform(minimum=1500, maximum=2500)

likelihood = tupak.GravitationalWaveTransient(interferometers=IFOs, waveform_generator=waveform_generator,
                                              time_marginalization=False, phase_marginalization=False,
                                              distance_marginalization=False, prior=priors)

result = tupak.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty', npoints=100,
                           injection_parameters=injection_parameters, outdir=outdir, label=label)

result.plot_corner(lionize=True)
print(result)
