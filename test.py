from tdscf import *
import pyscf, sys, re
import pyscf.dft
import numpy as np
from cmath import *
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

a = '''
C	-9.1359898096	   0.9100475486     -0.0124926202
O	-8.3947824255     -0.2842834621      0.1495678782
H	-8.3341119742     -0.7088045755     -0.7224749286
H	-9.2735413716	   1.3763476427      0.9658671365
H      -10.1132024998	   0.6806654917     -0.4446094308
H	-8.5869741155	   1.5925211614     -0.6654882142
O	-5.0336962149	   3.9357565465     -0.2193849657
H	-4.2806307762	   3.4326339246      0.1520368236
H	-5.6369690003	   3.9974245127      0.5371447273
O	-3.2183834942	   1.7512178171     -0.0679657106
H	-2.3795840928	   1.3328041938     -0.3234318405
H	-3.7618048367	   1.6656040442     -0.8706572153
O	-5.2143272326     -2.8202591585     -0.1271182603
H	-4.6634879763     -2.0313575154     -0.0035930872
H	-5.3549067783     -2.8700467270     -1.0820633770
    '''
prm = '''
Model  TDDFT
Method MMUT

dt	0.02
MaxIter 25000

ExDir   1.0
EyDir   1.0
EzDir   1.0
FieldAmplitude  0.001
FieldFreq       0.9202
ApplyImpulse    1
ApplyCw         0

StatusEvery     5000
'''
output = re.sub("py","dat",sys.argv[0])

bas = ['6-31g*','sto-3g']
xc = ['PBE0','PBE,PBE']

nA = 6
AA = False
if len(sys.argv) == 2:
        print "This is multi-threaded process. Thread Number:", sys.argv[1]
        bo1 = BORHF(a,bas,nA,xc,prm,output, AA,sys.argv[1])
elif len(sys.argv) == 1:
        print "This is single-threaded process."
        bo1 = BORHF(a,bas,nA,xc,prm,output, AA)
