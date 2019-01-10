import numpy as np

"""
Generate mock data
"""
def generate():
    telescope_id = 1
    machine_id = 10
    nchans = 128
    nbits = 4
    tstart = 50000.0
    tsamp = 80.0e-6
    fch1 = 433.968
    foff = -0.062
    nifs = 1
    period = 3.1415927
    width= 10
    dm = 30 
    tobs= 42 
    nbits = 4
    nsamples = period / tsamp
    nsblk = 512
    pulse = 0
    rwsum = 0

    ic = nifs*nchans
    arraysize = nchans*nifs*nsblk

    fblock = np.empty(arraysize, dtype=object)

    for s in range(0, nsblk):
        for i in range(0, nifs):
            for c in range(0, nchans):
                fblock[s*ic+i*nchans+c] = fch1 + c * foff
            
    fileStr = "HEADER_START"
    fileStr += "P:" + str(period) + " ms"
    fileStr += "DM:" + str(dm)
    fileStr += "data_type: 1"
    fileStr += "machine_id:" + str(machine_id)
    fileStr += "telescope_id:" + str(telescope_id)
    fileStr += "nchans:" + str(nchans)
    fileStr += "tstart:" + str(tstart)
    fileStr += "fch1:" + str(fch1)
    fileStr += "HEADER_END"
    fileStr += "SIGNAL_START"
    fileStr += ''.join(map(str, fblock))
    fileStr += "SIGNAL_END"

    file = open("../pspm.fil","w") 
 
    file.write(fileStr)
 
    file.close() 

generate()