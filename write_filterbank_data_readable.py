import filterbank.filterbank as filterbank
import filterbank.header as header
import filterbank.filterbank as filterbank

f= open("filReadable.txt","w+")

fb = filterbank.Filterbank(filename='./pspm32.fil', read_all=True)
f.write('header:            \n')
f.write(str(fb.get_header()) + '\n')
f.write('setup_channels:    \n')
f.write(str(fb.setup_chans()) + '\n')
f.write('number of channels:\n')
f.write(str(fb.n_chans) + '\n')
f.write('number of samples: \n')
f.write(str(fb.n_samples) + '\n')
f.write('frequencies:       \n')
i = 0
for freq in fb.get_freqs():
    f.write(str(i) + ': ' + str(freq) + '\n')
    i += 1
f.write('data:              \n')
i = 0
for data in fb.data:
    f.write(str(i) + ': ' + str(data) + '\n')
    i += 1

# doesnt work:
# for iterator in range(len(fb.get_freqs())):
#     f.write(str(fb.get_freqs()[iterator]) + '\n')
#     f.write(str(fb.data()[iterator]) + '\n')
