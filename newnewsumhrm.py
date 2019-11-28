

def newnewsumhrm(SP,NF,NF1,P1,P2,P4,P8,P16):
    # INTEGER NF, NF1
    # REAL SP(NF) list sp with size nf
    # INTEGER ifold
    # REAL fval
    # INTEGER NFS
    # REAL X, XDIV
    # INTEGER IDX, LSTIDX
    # INTEGER I, N, NFOLDS
    # INTEGER FOLDVALS(5) list FOLDVALS with size 5
    # REAL P1(*), P2(*), P4(*), P8(*), P16(*) lists with unknown size

    NFOLDS = 5
    FOLDVALS = [1,2,4,8,16]
    for ifold in range(NFOLDS):
        fval = FOLDVALS[ifold]
        if fval == 1:
            for n in range(NF):
                P1[n] = SP[n]
        if fval == 2:
            NFS = min(NF)
            XDIV = 1./fval
            for n in range(NFS, NF):
                P2[n] = 0
                LSTIDX = -1
                for i in range(1, FOLDVALS[ifold]):
                    X = n * i * XDIV + 0.5
                    IDX = X
                    if (IDX > 1) and (IDX != LSTIDX):
                        P2[n] = P2[n] + SP[IDX]
                    LSTIDX = IDX
        if fval == 4:
            NFS = min(NF)
            XDIV = 1./fval
            for n in range(NFS, NF):
                P4[n] = 0.
                LSTIDX = -1
                for i in range(1, FOLDVALS[ifold]):
                    X = n * i * XDIV + 0.5
                    IDX = X
                    if (IDX > 1) and (IDX != LSTIDX):
                        P4[n] = P4[n] + SP[IDX]
                    LSTIDX = IDX
        if fval == 8:
            NFS = min(NF)
            XDIV = 1. / fval
            for n in range(NFS, NF):
                P8[n] = 0.
                LSTIDX = -1
                for i in range(1, FOLDVALS[ifold]):
                    X = n * i * XDIV + 0.5
                    IDX = X
                    if (IDX > 1) and (IDX != LSTIDX):
                        P8[n] = P8[n] + SP[IDX]
                    LSTIDX = IDX
        if fval == 16:
            NFS = min(NF)
            XDIV = 1. / fval
            for n in range(NFS, NF):
                P16[n] = 0.
                LSTIDX = -1
                for i in range(1, FOLDVALS[ifold]):
                    X = n * i * XDIV + 0.5
                    IDX = X
                    if (IDX > 1) and (IDX != LSTIDX):
                        P16[n] = P16[n] + SP[IDX]
                    LSTIDX = IDX
        return
