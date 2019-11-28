from harmonic_summing import harmonic_summing as harmsum
import filterbank.filterbank as filterbank

fb = filterbank.Filterbank(filename='./pspm32.fil', read_all=True)
# get filertbank data + frequency labels
freqs, fil_data = fb.select_data()

harmsum.apply_harmonic_summing(frequencies=freqs, fil_data=fil_data, precision=0.001, num_lo_up_harmonics=(5, 5))

