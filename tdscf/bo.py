import numpy as np
import scipy, time
import scipy.linalg
from func import *
import func
from pyscf import gto, dft, scf, ao2mo
from tdfields import *
import ctypes

FsPerAu = 0.0241888

class BORHF():
    def __init__(self, atom, basis, n, xc = None, prm = None, output = 'log.dat', AA = True):
        """
        Make a BlockOrthogonalizedRHF
        Args: mol1_, mol2_ fragments to make a block orthogonal SCF from.
        """

        self.AA = AA
        self.adiis = None
        self.JKtilde = None
        self.basis = basis
        self.m = []
        self.scf = []
        self.hyb = [0,0,0]
        # objects
        self.m1, self.m2, self.m3 = self.MixedBasisMol(atom,basis,n)
        self.hf1, self.hf2, self.hf3 = self.MixedTheory(xc)
        # Global Variables
        self.AOExc = None
        self.BOExc = None
        self.EBO = None
        self.nA = self.m1.nao_nr()
        self.n = self.m3.nao_nr()
        self.n_ao = np.zeros(3)
        self.n_mo = None
        self.n_aux = np.zeros(3)
        self.n_occ = int(sum(self.hf3.mo_occ)/2)
        self.Enuc = self.hf3.energy_nuc()
        self.Exc = np.zeros(3)
        # Global Matrices
        self.C = None
        self.eri3c = []
        self.eri2c = []
        self.H = self.hf3.get_hcore()
        self.S = self.hf3.get_ovlp() # AOxAO
        self.U = BOprojector(self.m1,self.m3) #(AO->BO)
        self.Htilde = TransMat(self.H,self.U)
        self.Stilde = TransMat(self.S,self.U)
        self.X = MatrixPower(self.S,-1./2.) # AO->LAO
        self.Xtilde = MatrixPower(self.Stilde,-1./2.)
        self.O = None #(MO->BO) = C * U
        self.Vtilde = None

        # Propagation Steps
        #print self.hyb
        self.B0 = None # nA,nA, n_aux0
        self.B1 = None # n, n, n_aux1
        self.auxmol_set()
        self.CBOsetup()
        self.params = dict()
        self.initialcondition(prm)
        self.field = fields(self.hf3, self.params)
        self.field.InitializeExpectation(self.rho, self.C, self.nA)
        start = time.time()
        self.prop(output)
        end = time.time()
        print "Propagation time:", end - start
        return

    def CBOsetup(self):
        print "============================="
        libtdscf.SetupBO(\
        ctypes.c_int(int(self.nA)),ctypes.c_int(int(self.n_ao[2])),ctypes.c_int(int(self.n_occ)),ctypes.c_int(int(self.n_aux[2])),ctypes.c_int(int(self.n_aux[0])),\
        self.U.ctypes.data_as(ctypes.c_void_p), \
        self.B0.ctypes.data_as(ctypes.c_void_p), self.B1.ctypes.data_as(ctypes.c_void_p))

    def MixedBasisMol(self, atm, bas, n):
        # For mixed Basis of AA and BB
        p = 0
        n1 = 0
        n2 = 0
        atm1 = ''
        atm2 = ''
        set1 = []
        set2 = []

        for line in atm.splitlines():
            line1 = line.lstrip()
            if len(line1) > 1:
                if p < n:
                    atm1 += "@"+line1+"\n"
                    set1.append("@"+line1[0:1])
                    n1 += 1
                else:
                    atm2 += line1 + "\n"
                    set2.append(line1[0:1])
                    n2 += 1
                p += 1
        atm0 = atm1 + atm2
        bas12 = {}
        for i in range(n1):
            bas12[set1[i]] = bas[0]
        for i in range(n2):
            bas12[set2[i]] = bas[1]

        mol1 = gto.Mole()
        mol1.atom = atm1
        mol1.basis = bas[0]
        mol1.build()

        mol2 = mol1

        mol3 = gto.Mole()
        mol3.atom = atm0
        mol3.basis = bas12
        mol3.build()


        return mol1,mol2,mol3

    def MixedTheory(self,xc):
        # Generate Mixed HF objects
        print "\n================"
        print "=   AA Block   ="
        print "================"
        print "Basis:", self.basis[0]
        print "Theory:",xc[0]
        m1 = dft.rks.RKS(self.m1)
        m1.xc = xc[0]
        m1.grids.level = 1
        m1.kernel()
        self.scf.append(m1)
        self.m.append(self.m1)
        self.hyb[0] = m1._numint.hybrid_coeff(m1.xc, spin=(m1.mol.spin>0)+1)
        print "Basis:", self.basis[0]
        print "Theory:",xc[1]
        m2 = dft.rks.RKS(self.m2)
        m2.xc = xc[1]
        m2.grids.level = 1
        m2.kernel()
        self.scf.append(m2)
        self.m.append(self.m2)
        self.hyb[1] = m2._numint.hybrid_coeff(m2.xc, spin=(m2.mol.spin>0)+1)
        print "\n================="
        print "=  Whole Block  ="
        print "================="
        print "Basis:", self.basis[1]
        print "Theory:",xc[1]
        m3 = dft.rks.RKS(self.m3)
        m3.xc = xc[1]
        m3.grids.level = 1
        m3.kernel()
        self.scf.append(m3)
        self.m.append(self.m3)
        self.hyb[2] = m3._numint.hybrid_coeff(m3.xc, spin=(m3.mol.spin>0)+1)

        return m1,m2,m3


    def auxmol_set(self,auxbas = "weigend"):
        print "===================="
        print "GENERATING INTEGRALS"
        print "===================="
        auxmol = gto.Mole()
        auxmol.atom = self.hf1.mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        mol = self.hf1.mol
        nao = self.n_ao[0] = mol.nao_nr()
        naux = self.n_aux[0] = auxmol.nao_nr()
        atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
        eri3c = np.empty((nao,nao,naux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.getints_by_shell('cint3c2e_sph', shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di

        eri2c = np.empty((naux,naux))
        pk = 0
        for k in range(mol.nbas, mol.nbas+auxmol.nbas):
            pl = 0
            for l in range(mol.nbas, mol.nbas+auxmol.nbas):
                shls = (k, l)
                buf = gto.getints_by_shell('cint2c2e_sph', shls, atm, bas, env)
                dk, dl = buf.shape
                eri2c[pk:pk+dk,pl:pl+dl] = buf
                pl += dl
            pk += dk


        self.eri3c.append(eri3c)
        self.eri2c.append(eri2c)
        print "\nAA INT GENERATED"

        self.eri3c.append(eri3c)
        self.eri2c.append(eri2c)

        RSinv = MatrixPower(eri2c,-0.5)
        self.B0 = np.einsum('ijp,pq->ijq', eri3c, RSinv)

        auxmol = mol = nao = naux = None

        auxmol = gto.Mole()
        auxmol.atom = self.hf3.mol.atom
        auxmol.basis = auxbas
        auxmol.build()
        mol = self.hf3.mol
        nao = self.n_ao[2] = mol.nao_nr()
        naux = self.n_aux[2] = auxmol.nao_nr()
        atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env)
        eri3c = np.empty((nao,nao,naux))
        pi = 0
        for i in range(mol.nbas):
            pj = 0
            for j in range(mol.nbas):
                pk = 0
                for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                    shls = (i, j, k)
                    buf = gto.getints_by_shell('cint3c2e_sph', shls, atm, bas, env)
                    di, dj, dk = buf.shape
                    eri3c[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                    pk += dk
                pj += dj
            pi += di

        eri2c = np.empty((naux,naux))
        pk = 0
        for k in range(mol.nbas, mol.nbas+auxmol.nbas):
            pl = 0
            for l in range(mol.nbas, mol.nbas+auxmol.nbas):
                shls = (k, l)
                buf = gto.getints_by_shell('cint2c2e_sph', shls, atm, bas, env)
                dk, dl = buf.shape
                eri2c[pk:pk+dk,pl:pl+dl] = buf
                pl += dl
            pk += dk

        print "\nWHOLE INT GENERATED"
        self.eri3c.append(eri3c)
        self.eri2c.append(eri2c)
        RSinv = MatrixPower(eri2c,-0.5)
        self.B1 = np.einsum('ijp,pq->ijq', eri3c, RSinv)
        auxmol = mol = nao = naux = None
        return

    def initialcondition(self,prm):
        n_ao = self.n_ao[2] = self.hf3.mol.nao_nr()#self.hf3.make_rdm1().shape[0]
        n_mo = self.n_mo = n_ao # should be fixed.
        n_occ = self.n_occ = int(sum(self.hf3.mo_occ)/2)
        print "n_ao:", n_ao, "n_mo:", n_mo, "n_occ:", n_occ
        print "nA:",self.nA,"n:", self.n
        self.ReadParams(prm)
        self.InitializeLiouvillian()
        return

    def ReadParams(self,prm):
        self.params["Model"] = "TDDFT"
        self.params["Method"] = "MMUT"

        self.params["dt"] =  0.02
        self.params["MaxIter"] = 15000

        self.params["ExDir"] = 1.0
        self.params["EyDir"] = 1.0
        self.params["EzDir"] = 1.0
        self.params["FieldAmplitude"] = 0.01
        self.params["FieldFreq"] = 0.9202
        self.params["Tau"] = 0.07
        self.params["tOn"] = 7.0*self.params["Tau"]
        self.params["ApplyImpulse"] = 1
        self.params["ApplyCw"] = 0

        self.params["StatusEvery"] = 5000
        # Here they should be read from disk.
        if(prm != None):
            for line in prm.splitlines():
                s = line.split()
                if len(s) > 1:
                    if s[0] == "MaxIter" or s[0] == str("ApplyImpulse") or s[0] == str("ApplyCw") or s[0] == str("StatusEvery"):
                        self.params[s[0]] = int(s[1])
                    elif s[0] == "Model" or s[0] == "Method":
                        self.params[s[0]] = s[1]
                    else:
                        self.params[s[0]] = float(s[1])

        print "============================="
        print "         Parameters"
        print "============================="
        print "Model:", self.params["Model"]
        print "Method:", self.params["Method"]
        print "dt:", self.params["dt"]
        print "MaxIter:", self.params["MaxIter"]
        print "ExDir:", self.params["ExDir"]
        print "EyDir:", self.params["EyDir"]
        print "EzDir:", self.params["EzDir"]
        print "FieldAmplitude:", self.params["FieldAmplitude"]
        print "FieldFreq:", self.params["FieldFreq"]
        print "Tau:", self.params["Tau"]
        print "tOn:", self.params["tOn"]
        print "ApplyImpulse:", self.params["ApplyImpulse"]
        print "ApplyCw:", self.params["ApplyCw"]
        print "StatusEvery:", self.params["StatusEvery"]
        print "=============================\n\n"
        return

    def InitializeLiouvillian(self):
        self.InitFockBuild()
        self.rho = 0.5*np.diag(self.hf3.mo_occ).astype(complex)
        self.rhoM12 = self.rho.copy()

    def InitFockBuild(self):
        '''
        Using Roothan's equation to build a Fock matrix and initial density matrix
        Returns:
            self consistent density in Block Orthogonalized basis.
        '''
        nA = self.nA
        n = self.n
        n_occ = self.n_occ
        Ne = self.n_e = 2.0 * n_occ
        noc = int(Ne/2)
        err = 100
        it = 0
        dm = self.hf3.make_rdm1()
        Uinv = np.linalg.inv(self.U)
        Pbo = 0.5 * TransMat(dm, Uinv, -1)
        print dm - 2.0 * TransMat(Pbo,self.U,-1)
        print "Ne (AO):", TrDot(dm, self.S)
        print "Ne (BO):", TrDot(Pbo,self.Stilde)

        adiis = self.hf3.DIIS(self.hf3, self.hf3.diis_file)
        adiis.space = self.hf3.diis_space
        adiis.rollback = self.hf3.diis_space_rollback
        self.adiis = adiis

        self.F = self.BO_Fockbuild(Pbo)
        E = self.energy(Pbo)
        print "Initial Energy:",E
        e_conv = 10**-7
        while (err > e_conv):
            # Diagonalize F in the BO basis
            self.eigs, Ctilde = scipy.linalg.eigh(self.F,self.Stilde)
            occs = np.ones(self.n,dtype = np.complex)
            occs[noc:] *= 0.0
            Pmo = np.diag(occs).astype(complex)
            Pbo = TransMat(Pmo,Ctilde,-1)
            print "NE(Lit):", TrDot(Pbo,self.Stilde)
            Eold = E
            self.F = self.BO_Fockbuild(Pbo,it) # in BO Basis
            E = self.energy(Pbo)
            err = abs(E-Eold)
            if (it%1 ==0):
                print "Iteration:", it,"; Energy:",E,"; Error =",err
            it += 1
            if(it == 50):
                print "Lowering Error limit to 10^-6"
                e_conv = 10**-6
            elif(it == 100):
                print "Lowering Error limit to 10^-5"
                e_conv = 10**-5
            elif(it == 200):
                print "SCF Convergence not met"
                break
        self.Ctrev = np.dot(self.Stilde,Ctilde)
        self.C = np.dot(self.U, Ctilde) #|AO><BO|
        Pmo = 0.5*np.diag(self.hf3.mo_occ).astype(complex)
        Pbo = TransMat(Pmo,Ctilde,-1)
        self.Ctilde = Ctilde.copy()
        self.rho = TransMat(Pbo,self.Ctrev)# BO to MO
        self.rhoM12 = TransMat(Pbo,self.Ctrev)
        print "Ne", TrDot(Pbo,self.Stilde), np.trace(self.rho),
        print "Energy:",E
        print "Initial Fock Matrix(MO)\n", TransMat(self.F,Ctilde)
        print self.eigs
        return Pbo


    def BO_Fockbuild(self,P,it = -1):
        """
        Updates self.F given current self.rho (both complex.)
        Fock matrix with HF
        Args:
            P = BO density matrix.
        Returns:
            Fock matrix(BO) . Updates self.Jtilde, Ktilde, JKtilde
        """
        nA = self.nA
        Pt = 2.0*TransMat(P,self.U,-1)
        PtA = 2.0*P[:nA,:nA]
        if self.params["Model"] == "TDDFT":
            J = self.get_j_c(Pt,int(2))
            self.Jtilde = TransMat(J,self.U)
            Veff = self.get_vxc(Pt,int(2))
            Veff = TransMat(Veff,self.U)
            Veff[:nA,:nA] += self.get_vxc(PtA,int(0))
            Veff[:nA,:nA] -= self.get_vxc(PtA,int(1))
            if(self.hyb[0] > 0.01):
                self.Ktilde = np.zeros((self.n,self.n)).astype(complex)
                Ktilde = self.get_k_c(PtA,int(0))
                self.Ktilde[:nA,:nA] = Ktilde
                Veff[:nA,:nA] += -0.5 * self.hyb[0] * self.Ktilde[:nA,:nA]
            JKtilde = self.Jtilde + Veff
            if self.adiis and it > 0:
                return self.adiis.update(self.Stilde,2*P,self.Htilde + 0.5 * (JKtilde + JKtilde.T.conj()))
            else:
                return self.Htilde + 0.5 * (JKtilde + JKtilde.T.conj())


    def get_vxc(self,P,mol = 3):
        '''
        Args:
            P: BO density matrix

        Returns:
            Vxc: Exchange and Correlation matrix (AO)
        '''
        Vxc = self.numint_vxc(self.scf[mol]._numint,P,mol)
        return Vxc


    def energy(self,Pbo,IfPrint=False):
        nA = self.nA
        n = self.n
        if self.params["Model"] == "TDDFT":
            EH = np.trace(np.dot(Pbo,2.0 * self.Htilde))
            EJ = np.trace(np.dot(Pbo,1.0 * self.Jtilde))
            EK = 0
            # Under the assumption that low theory will not use hybrid functional
            if(self.hyb[0] > 0.01):
                EK = self.hyb[0] * TrDot(Pbo, -0.5 * self.Ktilde)
            Exc = self.Exc[2]
            Exch = self.Exc[0]
            Excl = self.Exc[1]
            E = EH + EJ + EK + Exc + Exch - Excl + self.Enuc
            return E.real

    def numint_vxc(self,ni,P,mol = 3,max_mem = 2000):
        xctype = self.scf[mol]._numint._xc_type(self.scf[mol].xc)
        make_rho, nset, nao = self._gen_rho_evaluator(self.m[mol], P, 1)
        ngrids = len(self.scf[mol].grids.weights)
        non0tab = self.scf[mol]._numint.non0tab
        vmat = np.zeros((nset,nao,nao)).astype(complex)
        excsum = np.zeros(nset)
        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(self.m[mol], self.scf[mol].grids, nao, ao_deriv, max_mem, non0tab):
                rho = make_rho(0, ao, mask, 'LDA')
                exc, vxc = ni.eval_xc(self.scf[mol].xc, rho, 0, 0, 1, None)[:2]
                vrho = vxc[0]
                den = rho * weight
                excsum[0] += (den * exc).sum()
                aow = np.einsum('pi,p->pi', ao, .5*weight*vrho)
                vmat += func._dot_ao_ao(self.m[mol], ao, aow, nao, weight.size, mask)
                rho = exc = vxc = vrho = aow = None
        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(self.m[mol], self.scf[mol].grids, nao, ao_deriv, max_mem, non0tab):
                ngrid = weight.size
                rho = make_rho(0, ao, mask, 'GGA')
                exc, vxc = ni.eval_xc(self.scf[mol].xc, rho, 0, 0, 1, None)[:2]
                vrho, vsigma = vxc[:2]
                wv = np.empty((4,ngrid))#.astype(complex)
                wv[0]  = weight * vrho * .5
                wv[1:] = rho[1:] * (weight * vsigma * 2)
                aow = np.einsum('npi,np->pi', ao, wv)
                vmat += func._dot_ao_ao(self.m[mol], ao[0], aow, nao, ngrid, mask)
                den = rho[0] * weight
                excsum[0] += (den * exc).sum()
                rho = exc = vxc = vrho = vsigma = wv = aow = None
        else:
            assert(all(x not in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')))
            ao_deriv = 2
            for ao, mask, weight, coords in ni.block_loop(self.m[mol], self.scf[mol].grids, nao, ao_deriv, max_mem, non0tab):
                ngrid = weight.size
                rho = make_rho(0, ao, mask, 'MGGA')
                exc, vxc = ni.eval_xc(self.scf[mol].xc, rho, 0, 0, 1, None)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den = rho[0] * weight
                excsum[0] += (den * exc).sum()
                wv = np.empty((4,ngrid))
                wv[0]  = weight * vrho * .5
                wv[1:] = rho[1:4] * (weight * vsigma * 2)
                aow = np.einsum('npi,np->pi', ao[:4], wv)
                vmat += func._dot_ao_ao(self.m[mol], ao[0], aow, nao, ngrid, mask)
                wv = (.5 * .5 * weight * vtau).reshape(-1,1)
                vmat += func._dot_ao_ao(self.m[mol], ao[1], wv*ao[1], nao, ngrid, mask)
                vmat += func._dot_ao_ao(self.m[mol], ao[2], wv*ao[2], nao, ngrid, mask)
                vmat += func._dot_ao_ao(self.m[mol], ao[3], wv*ao[3], nao, ngrid, mask)
                rho = exc = vxc = vrho = vsigma = wv = aow = None
        self.Exc[mol] = excsum[0]
        Vxc = vmat.reshape(nao,nao)
        Vxc = Vxc + Vxc.T.conj()
        return Vxc

    def _gen_rho_evaluator(self, mol, dms, hermi=1):
        natocc = []
        natorb = []
        e, c = scipy.linalg.eigh(dms)
        natocc.append(e)
        natorb.append(c)
        nao = dms.shape[0]
        ndms = len(natocc)
        def make_rho(idm, ao, non0tab, xctype):
            return eval_rhoc(mol, ao, natorb[idm], natocc[idm], non0tab, xctype)
        return make_rho, ndms, nao

    def get_jk_c(self, P, mol = 3):
        '''
        Args:
            P: AO density matrix
        Returns:
            J: Coulomb matrix
            K: Exchange matrix
        '''
        return self.get_j_c(P,mol), self.get_k_c(P,mol)

    def get_j_c(self, P, mol = 3):
        '''
        Args:
            P: AO density matrix

        Returns:
            J: Coulomb matrix (AO)
        '''
        if mol == 2:
            Pc = np.asarray(P, order='C').astype(complex)
            Jmat = np.zeros((self.n,self.n))
            libtdscf.get_j1(\
            Pc.ctypes.data_as(ctypes.c_void_p), Jmat.ctypes.data_as(ctypes.c_void_p))
        return Jmat

    def get_k_c(self, P, mol = 3):
        '''
        Args:
            P: BO density matrix
        Returns:
            K: Exchange matrix (BO)
        '''
        if mol == 0:
            Pc = np.asarray(P, order='C').astype(complex)
            Kmat = np.zeros((self.nA,self.nA)).astype(complex)
            libtdscf.get_k0(\
            Pc.ctypes.data_as(ctypes.c_void_p), Kmat.ctypes.data_as(ctypes.c_void_p))
        return Kmat

    def Step_MMUT(self, w, v , oldrho , time, dt ,IsOn):
        Ud = np.exp(w*(-0.5j)*dt);
        U = TransMat(np.diag(Ud),v,-1)
        RhoHalfStepped = TransMat(oldrho,U,-1)
        newrho = TransMat(RhoHalfStepped,U,-1)
        return newrho

    def TDDFTstep(self,time):
        self.F = self.BO_Fockbuild(TransMat(self.rho,self.Ctilde,-1)) # BO basis
        self.F = np.conj(self.F)
        Fmo_prev = TransMat(self.F,self.Ctilde) # to MO basis

        self.eigs, rot = np.linalg.eig(Fmo_prev)
        self.rho = TransMat(self.rho, rot)

        self.rhoM12 = TransMat(self.rhoM12, rot)
        Fmo = np.diag(self.eigs).astype(complex)
        self.Ctilde = np.dot(self.Ctilde,rot)
        self.C = np.dot(self.C, rot)
        FmoPlusField, IsOn = self.field.ApplyField(Fmo,self.Ctilde,time)
        w,v = scipy.linalg.eig(FmoPlusField)
        NewRhoM12 = self.Step_MMUT(w, v, self.rhoM12, time, self.params["dt"], IsOn)
        NewRho = self.Step_MMUT(w, v, NewRhoM12, time,self.params["dt"]/2.0, IsOn)
        self.rho = 0.5*(NewRho+(NewRho.T.conj()));

        self.rhoM12 = 0.5*(NewRhoM12+(NewRhoM12.T.conj()))

    def dipole(self, AA = False):
        if AA == False:
            return self.field.Expectation(self.rho, self.C, AA)
        else:
            return self.field.Expectation(self.rho, self.C, AA, self.nA,self.U)
    def loginstant(self,iter):
        """
        time is logged in atomic units.
        """
        np.set_printoptions(precision = 7)
        tore = str(self.t)+" "+str(self.dipole().real).rstrip(']').lstrip('[')+ " " +str(self.energy(TransMat(self.rho,self.Ctilde,-1),False))+" "+str(np.trace(self.rho))
        if iter%self.params["StatusEvery"] ==0 or iter == self.params["MaxIter"]-1:
            print 't:', self.t*FsPerAu, " (Fs)  Energy:",self.energy(TransMat(self.rho,self.Ctilde,-1)), " Tr ",(np.trace(self.rho))
            print('Dipole moment(X, Y, Z, au): %8.5f, %8.5f, %8.5f' %(self.dipole().real[0],self.dipole().real[1],self.dipole().real[2]))
        return tore

    def AAlog(self,iter):
        """
        time is logged in atomic units.
        """
        np.set_printoptions(precision = 7)
        tore = str(self.t)+" "+str(self.dipole(True).real).rstrip(']').lstrip('[')
        return tore


    def step(self,time):
        """
        Performs a step
        Updates t, rho, and possibly other things.
        """
        return self.TDDFTstep(time)



    def prop(self,output):
        """
        The main tdscf propagation loop.
        """
        iter = 0
        self.t = 0
        f = open(output,'a')
        start = time.time()
        aa = open(output + '.aa','a')
        print "Energy Gap (eV)",abs(self.eigs[self.n_occ]-self.eigs[self.n_occ-1])*27.2114
        print "\n\nPropagation Begins"
        while (iter<self.params["MaxIter"]):
            self.step(self.t)
            f.write(self.loginstant(iter)+"\n")
            if (self.AA == True):
                aa.write(self.AAlog(iter) + "\n")

            self.t = self.t + self.params["dt"]
            if iter%self.params["StatusEvery"] ==0 or iter == self.params["MaxIter"]-1:
                end = time.time()
                print (end - start)/(60*60*self.t * FsPerAu * 0.001), "hr/ps"
            iter = iter + 1
        f.close()
        aa.close()





def BOprojector(mol1,mol2):
    S = scf.hf.get_ovlp(mol2)
    nA = gto.nao_nr(mol1)
    nAB = gto.nao_nr(mol2)
    nB = nAB - nA
    SAA = S[0:nA, 0:nA]
    SBB = S[nA:nAB, nA:nAB]
    SAB = S[0:nA,nA:nAB]
    U = np.identity(nAB)
    PAB = np.dot(np.linalg.inv(SAA), SAB)
    U[0:nA,nA:nAB] = -PAB
    return U

def z2cart(a):
    acart = gto.mole.from_zmatrix(a)
    b = ''
    for x in acart:
        coord = str(x[1])
        coord = coord.rstrip(']')
        coord = coord.lstrip('[')
        b += x[0] + " " + coord + "\n"
    return b
