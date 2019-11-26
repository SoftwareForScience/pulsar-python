from harmonic_summing import harmonic_summing as harmsum
import filterbank.filterbank as filterbank
import plot as plt
fb = filterbank.Filterbank(filename='./pspm32.fil', read_all=True)
# get filertbank data + frequency labels
fil_data = fb.select_data()

harmsum = harmsum.apply_harmonic_summing(fil_data=fil_data)
print(harmsum)
#
#
# # Plot the PSD
# plt.grid(True)
# plt.xlabel('Frequency (MHz)')
# plt.ylabel('Intensity (arbitrary units)')
# plt.plot(f, power_levels)
# plt.show()
