def readFile():
    f = '../sigproc/sigproc/src/pspm.fil'
    with open(f, "r") as ins:
        array = []
        for line in ins:
            array.append(line)
    print(array[:16])

readFile()