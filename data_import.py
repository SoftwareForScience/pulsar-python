import filterbank.filterbank as fb

# Download the files in the README.md

def _8bit(freq_start, freq_stop):
    fb1 = fb.Filterbank(filename = "../data/pspm8.fil")

    test1 = fb1.select_data(freq_start, freq_stop)

    return test1


def _16bit(freq_start, freq_stop):
    fb2 = fb.Filterbank(filename = '../data/pspm16.fil')

    test2 = fb2.select_data(freq_start, freq_stop)

    return test2

def _32bit(freq_start, freq_stop):
    fb3 = fb.Filterbank(filename = '../data/pspm32.fil')

    test3 = fb3.select_data(freq_start, freq_stop)

    return test3
