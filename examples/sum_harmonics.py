from harmonic_summing import harmonic_summing as harmsum
import filterbank.filterbank as filterbank

fb = filterbank.Filterbank(filename='./pspm32.fil', read_all=True)
# get filertbank data + frequency labels
fil_data = fb.select_data()

harmsum.apply_harmonic_summing(fil_data=fil_data)

