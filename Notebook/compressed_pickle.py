import pickle
import bz2

def loads(comp):
    return pickle.loads(bz2.decompress(comp))

def dumps(obj):
    return bz2.decompress(pickle.dumps(obj))

def load(fname):
    fin = bz2.BZ2File(fname, 'rb')
    try:
        pkl = fin.read()
    finally:
        fin.close()
    return pickle.loads(pkl)

def dump(obj, fname, level=9):
    pkl = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    fout = bz2.BZ2File(fname, 'wb', compresslevel=level)
    try:
        fout.write(pkl)
    finally:
        fout.close()