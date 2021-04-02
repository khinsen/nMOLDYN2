from Numeric import *
from FFT import fft,inverse_fft
from MMTK import *
from MMTK.Trajectory import Trajectory
from MMTK.ParticleProperties import ParticleVector, Configuration
from MMTK.Random import randomDirection
from Scientific.Geometry import Vector, Tensor
from random import randint, uniform
import Numeric
#import cephes

# Use FFTW if available, FFTPACK otherwise. In older releases, the
# FFTW module is called dfftw.

try:
    import fftw
except ImportError:
    fftw = None

if fftw is None:
    try:
        import dfftw
        fftw = dfftw
    except ImportError:
        pass

if fftw is None:
    import FFT

a2 = array([[   -3.,   4.,  -1.],
            [   -1.,   0.,   1.],
            [    1.,  -4.,   3.]])
a3 = array([[  -11.,  18.,  -9.,   2.],
            [   -2.,  -3.,   6.,  -1.],
            [    1.,  -6.,   3.,   2.],
            [   -2.,   9., -18.,  11.]])
a4 = array([[  -50.,  96., -72.,  32.,  -6.],
            [   -6., -20.,  36., -12.,   2.],
            [    2., -16.,   0.,  16.,  -2.],
            [   -2.,  12., -36.,  20.,   6.],
            [    6., -32.,  72., -96.,  50.]])
a5 = array([[  -274.,  600., -600.,  400., -150.,  24.],
            [   -24., -130.,  240., -120.,   40.,  -6.],
            [     6.,  -60.,  -40.,  120.,  -30.,   4.],
            [    -4.,   30., -120.,   40.,   60.,  -6.],
            [     6.,  -40.,  120., -240.,  130.,  24.],
            [   -24.,  150., -400.,  600., -600., 274.]])


if fftw is None:

    def fft(series, lock=None):
        return FFT.fft(series)

    def inverse_fft(series, lock=None):
        return FFT.inverse_fft(series)

    def acf(series, padding=1, lock=None):
        if padding:
            n = 1
            while n < 2*len(series):
                n = n*2    
        else:
            n = 2*len(series)
        FFTSeries = FFT.fft(series,n,0)
        FFTSeries = FFTSeries*conjugate(FFTSeries)
        FFTSeries = FFT.inverse_fft(FFTSeries,len(FFTSeries),0)
        return FFTSeries.real[:len(series)]

    def CorrelationFunction(series1, series2, lock=None):
        n = 1
        while n < 2*len(series1):
            n = n*2
        FFTSeries1 = FFT.fft(series1,n,0)
        FFTSeries2 = FFT.fft(series2,n,0)
        FFTSeries1 = conjugate(FFTSeries1)*FFTSeries2
        FFTSeries1 = FFT.inverse_fft(FFTSeries1,len(FFTSeries1),0)
        return FFTSeries1.real[:len(series1)] / \
                  (len(series1)-arange(len(series1)))

else:

    fftw_plans = {}

    def get_fftw_plans(shape, lock):
        if lock is not None: lock.acquire()
        try:
            plan_f, plan_b = fftw_plans[shape]
        except KeyError:
            dummy1 = Numeric.zeros(shape, Numeric.Complex)
            dummy2 = Numeric.zeros(shape, Numeric.Complex)
            plan_f = fftw.fftw_create_plan_specific(shape[0],
                                                    fftw.FFTW_FORWARD,
                                                    fftw.FFTW_THREADSAFE,
                                                    dummy1, 1, dummy2, 1)
            plan_b = fftw.fftw_create_plan_specific(shape[0],
                                                    fftw.FFTW_BACKWARD,
                                                    fftw.FFTW_THREADSAFE,
                                                    dummy1, 1, dummy2, 1)
            fftw_plans[shape] = plan_f, plan_b
        if lock is not None: lock.release()
        return plan_f, plan_b

    def fft(series, lock=None):
        shape = series.shape
        plan_f, plan_b = get_fftw_plans(shape, lock)
        temp = Numeric.zeros(shape, Numeric.Complex)    
        if len(shape) == 1:
            nseries = 1
        else:
            nseries = shape[1]
        fftw.fftw(plan_f, nseries, series+0.j, nseries, 1,
                  temp, nseries, 1)
        return temp

    def inverse_fft(series, lock=None):
        shape = series.shape
        plan_f, plan_b = get_fftw_plans(shape, lock)
        temp = Numeric.zeros(shape, Numeric.Complex)    
        if len(shape) == 1:
            nseries = 1
        else:
            nseries = shape[1]
        fftw.fftw(plan_b, nseries, series+0.j, nseries, 1,
                  temp, nseries, 1)
        return temp/shape[0]

    def acf(series, padding=1, lock=None):
        if padding:
            n = 1
            while n < 2*len(series):
                n = n*2    
        else:
            n = 2*len(series)
        shape = (n,) + series.shape[1:]
        plan_f, plan_b = get_fftw_plans(shape, lock)

        array1 = Numeric.zeros(shape, Numeric.Complex)
        array2 = Numeric.zeros(shape, Numeric.Complex)
        if len(shape) == 1:
            nseries = 1
        else:
            nseries = shape[1]
        array1[:len(series), ...] = series+0j
        fftw.fftw(plan_f, nseries, array1, nseries, 1,
                  array2, nseries, 1)
        Numeric.multiply(array2, Numeric.conjugate(array2), array2)
        fftw.fftw(plan_b, nseries, array2, nseries, 1,
                  array1, nseries, 1)

        return array1[:len(series)].real/n

    def CorrelationFunction(series1, series2, lock=None):
        n = 1
        while n < 2*len(series1):
            n = n*2
        shape = (n,) + series1.shape[1:]
        plan_f, plan_b = get_fftw_plans(shape, lock)

        array1 = Numeric.zeros(shape, Numeric.Complex)
        array2 = Numeric.zeros(shape, Numeric.Complex)
        array3 = Numeric.zeros(shape, Numeric.Complex)
        if len(shape) == 1:
            nseries = 1
        else:
            nseries = shape[1]
        array1[:len(series1), ...] = series1+0j
        fftw.fftw(plan_f, nseries, array1, nseries, 1,
                  array2, nseries, 1)
        array1[:] = 0.
        array1[:len(series2), ...] = series2+0j
        fftw.fftw(plan_f, nseries, array1, nseries, 1,
                  array3, nseries, 1)
        Numeric.multiply(array3, Numeric.conjugate(array2), array3)
        fftw.fftw(plan_b, nseries, array3, nseries, 1,
                  array1, nseries, 1)

        return array1[:len(series1)].real / \
               (float(n)*(len(series1)-Numeric.arange(len(series1))))


def AutoCorrelationFunction(series):
    """
    Autocorrelation function calculated by FFT method
    """
    if len(series.shape) == 1:
        return acf(series)/(len(series)-arange(len(series)))
    else:
        return acf(series)/(len(series)-arange(len(series)))[:,NewAxis]


def getMeanSquareDisplacement(series):
 
    MSD = zeros((len(series)),Float)
    dsq = series[:,0]**2+series[:,1]**2+series[:,2]**2
    sum_dsq1 = add.accumulate(dsq)
    sum_dsq2 = add.accumulate(dsq[::-1])
    sumsq = 2.*sum_dsq1[-1]
    msd = sumsq - concatenate(([0.], sum_dsq1[:-1])) \
                - concatenate(([0.], sum_dsq2[:-1]))
    Sab = 2.*add.reduce(acf(series,0), 1)
    MSD = MSD + (msd-Sab)/(len(series)-arange(len(series)))
    return  MSD

def GaussianWindow(series,alpha=0.):
 
    series1=zeros((2*len(series)-2,),Float)
    res = series*exp(-0.5*(alpha*arange(len(series))/(len(series)-1))**2)
    series1[:len(series)] = res
    series1[len(series):] = res[-2:0:-1]
    return series1

def getAngularVelocity(traj,group,timeInfo,reference=None,diffScheme='fast'):
    """ Calculation of Angular Velocity Autocorrelation Function
    from trajectory for a given group """
    
    time = traj.time[(timeInfo[0]):(timeInfo[1]):(timeInfo[2])]
    dt   = time[1] - time[0]
    rbt = traj.readRigidBodyTrajectory(object=group,
               first=timeInfo[0],last=timeInfo[1],
               skip=timeInfo[2],reference=reference)
    quaternions = rbt.quaternions
    quaternions_dot = zeros((len(quaternions),4),Float)
    for i in range(4):
        quaternions_dot[:,i] = diff(quaternions[:,i],diffScheme,dt)
    q = qMatrix(quaternions)
    angvel = 2*add.reduce(q*quaternions_dot[:,NewAxis,:], -1)
    return angvel[:,1:]


def qMatrix(data):
    """ transform a quaternion from the (scalar,vector) representation
    into the matrix form performing some rearrangements """
    
    res = zeros((len(data),4,4),Float)
    res[:,0,0]=  data[:,0] # qs       
    res[:,1,1]=  data[:,0]
    res[:,2,2]=  data[:,0]
    res[:,3,3]=  data[:,0]
    res[:,0,1]=  data[:,1] # qx
    res[:,1,0]= -data[:,1]
    res[:,2,3]=  data[:,1]
    res[:,3,2]= -data[:,1]
    res[:,0,2]=  data[:,2] # qy
    res[:,1,3]= -data[:,2]
    res[:,2,0]= -data[:,2]
    res[:,3,1]=  data[:,2]  
    res[:,0,3]=  data[:,3] # qz
    res[:,1,2]=  data[:,3]
    res[:,2,1]= -data[:,3]
    res[:,3,0]= -data[:,3]
    return res


def sphericalHarmonics(traj,jmn,group,refGroup,timeInfo):
 
    j, m, n = jmn
    quat = traj.readRigidBodyTrajectory(object=Collection(group),
                first=timeInfo[0],last=timeInfo[1],skip=timeInfo[2],
                reference=refGroup).quaternions
    c1  = ((2*jmn[0]+1)/(4*pi))**0.5
    quat2      = zeros(quat.shape,Numeric.Complex)  
    quat2[:,0] = quat[:,0]+1j*quat[:,3]
    quat2[:,2] = quat[:,2]+1j*quat[:, 1]
    quat2[:,1] = conjugate(quat2[:,0])
    quat2[:,3] = conjugate(quat2[:,2])
 
    if m == n:
        pp   = preparePP(j, m, n)
        res  = add.reduce(multiply.reduce(quat2[:,NewAxis,:]**pp[0][NewAxis,:,:],-1)*pp[1],1)
        res2 = res
    else:
        pp1  = preparePP(j, m, n)
        pp2  = preparePP(j, n, m)
        res  = add.reduce(multiply.reduce\
                 (quat2[:,NewAxis,:]**pp1[0][NewAxis,:,:],-1)*pp1[1],1)
        res2 = add.reduce(multiply.reduce\
                 (quat2[:,NewAxis,:]**pp2[0][NewAxis,:,:],-1)*pp2[1],1)
    return res*c1,res2*c1


def preparePP(j, m, n):
 
    c2 = (factorial(j+m) * factorial(j-m))**0.5 * factorial(j)
    c3 = j+m
    pp = []
    aa = []
    for p in range(0,j+1,1):
        if p <= c3 and p >= m:
            p1 = c3-p
            p2 = j-p
            p3 = p-m
            a1 = ((-1)**p)*c2/(factorial(p1)*factorial(p2)*\
                               factorial(p)*factorial(p3))
            pp.append((p1,p2,p3,p))
            aa.append(a1)
    return array(pp),array(aa)


def factorial(n):
    f = 1.
    while n > 0:
        f = f * n
        n = n - 1
    return f


def linearRegression(x,y):
    """
    fitting y = b + a*x
    """
    ss = len(x)
    sx = add.reduce(x)
    sy = add.reduce(y)
    sxoss = sx/ss
    t = x - sxoss
    st = add.reduce(t)
    st2 = add.reduce(t*t)
    b = add.reduce(t*y)/st2
    a = (sy-sx*b)/ss
    siga = sqrt((1+sx*sx/(ss*st2))/ss)
    sigb = sqrt(1.0/st2)
    covab = - sx/(ss*st2)
    rab = covab/(siga*sigb)
    chi2 = sqrt(sy - a - b*sx)
    fit = [a,b,siga,sigb,rab,chi2]
    return fit


def qVectorGenerator(qdata,traj):

    Qvalues, Qwidth, Qcount, Qdir = (qdata + (None,))[:4]
    Qlengths = []
    Qvectors = []
    if Qdir is None:
        for q in Qvalues:
            a = KVectors(traj.universe, q-0.5*Qwidth, q+0.5*Qwidth, Qcount)
            if len(a)==0:
                pass
            else:
                Qvectors.append(a)
                Qlengths.append(q)
    else:
        nv = Vector(Qdir).normal()
        for q in Qvalues:
            a = KVectors(traj.universe, None, None, None, q*nv)
            Qvectors.append(a)
            Qlengths.append(a[0].length())
    return Qlengths, Qvectors


class KVectors:

    def __init__(self, universe, lower, upper, number, initial=None):
        basis = universe.reciprocalBasisVectors()
        if basis is None:
            self.basis = None
            if initial is None:
                self._makeRandomKList(lower, upper, number)
            else:
                self.klist = [initial]
        else:
            self.basis = map(lambda v: 2.*pi*v, basis)
            inverse = array(map(list, universe.basisVectors()))
            self.inverse = Tensor(inverse)/(2.*pi)
            cell_volume = self.basis[0]*(self.basis[1].cross(self.basis[2]))
            if initial is None:
                center = 0.5*(lower+upper)
                shell_volume = 4.*pi*center**2 * (upper-lower)
                ncells = shell_volume/cell_volume
                if number > 1.5*ncells:
                    number = int(1.5*ncells)
                if ncells > 500:
                    self._makeRandomKList(lower, upper, number)
                else:
                    self._makeExplicitKList(lower, upper, number)
            else:
                ki = map(round, self.inverse*initial)
                k = ki[0]*self.basis[0] + ki[1]*self.basis[1] + \
                    ki[2]*self.basis[2]
                self.klist = [k]

    def _makeExplicitKList(self, lower, upper, number):
        upper_limit = map(lambda v, u=upper: int(u/v.length()), self.basis)
        klist = []
        for l in range(-upper_limit[0], upper_limit[0]+1):
            for m in range(-upper_limit[1], upper_limit[1]+1):
                for n in range(-upper_limit[2], upper_limit[2]+1):
                    k = l*self.basis[0] + m*self.basis[1] + n*self.basis[2]
                    if lower < k.length() < upper:
                        klist.append(k)
        if len(klist) <= number:
            self.klist = klist
        else:
            self.klist = []
            for i in range(number):
                self.klist.append(klist[randint(0, number-1)])

    def _makeRandomKList(self, lower, upper, number):
        self.klist = []
        while len(self.klist) < number:
            k = randomDirection()*uniform(lower, upper)
            if self.basis is not None:
                ki = map(round, self.inverse*k)
                k = ki[0]*self.basis[0] + ki[1]*self.basis[1] + \
                    ki[2]*self.basis[2]
            if lower < k.length() < upper:
                self.klist.append(k)

    def __len__(self):
        return len(self.klist)

    def __getitem__(self, index):
        return self.klist[index]


def getGroupTypes(traj,keys):

    universe = traj.universe
    nProtein = 0
    for object in universe:
        if object.__class__.__name__ == 'Protein':
           nProtein = nProtein+1
           name = 'Protein '+nProtein
           keep = Collection()
           groupType[name] = keep
           keep.addObject(object)
        else:
            pass


def diff(signal,diffScheme,dt):
    """ Differentiate a discrete input signal with
    different algorithms"""
    if diffScheme == 'fast':
        fact = 1./(2.*dt)
        result = zeros(len(signal),'d')
        result[0] = add.reduce(a2[0,:]*signal[:3])*fact
        gj = (signal[1:]-signal[:-1])/dt
        result[1:-1] = (gj[1:]+gj[:-1])/2.
        result[-1] = add.reduce(a2[2,:]*signal[-3:])*fact
    elif diffScheme == 'order 2':
        fact = 1./(2.*dt)
        result = zeros(len(signal),'d')
        result[0] = add.reduce(a2[0,:]*signal[:3])*fact
        gj = zeros((len(result)-2,3),'d')
        gj[:,0]   = a2[1,0]*signal[:-2]
        gj[:,1]   = a2[1,1]*signal[1:-1]
        gj[:,2]   = a2[1,2]*signal[2:]
        result[1:-1] = add.reduce(gj,-1)*fact
        result[-1] = add.reduce(a2[2,:]*signal[-3:])*fact
    elif diffScheme == 'order 3':
        fact = 1./(6.*dt)
        result = zeros(len(signal),'d')
        result[0] = add.reduce(a3[0,:]*signal[:4])*fact
        result[1] = add.reduce(a3[1,:]*signal[:4])*fact
        gj = zeros((len(result)-3,4),'d')
        gj[:,0]   = a3[2,0]*signal[:-3]
        gj[:,1]   = a3[2,1]*signal[1:-2]
        gj[:,2]   = a3[2,2]*signal[2:-1]
        gj[:,3]   = a3[2,3]*signal[3:]
        result[2:-1] = add.reduce(gj,-1)*fact
        result[-1] = add.reduce(a3[3,:]*signal[-4:])*fact
    elif diffScheme == 'order 4':
        fact = 1./(24.*dt)
        result = zeros(len(signal),'d')
        result[0] = add.reduce(a4[0,:]*signal[:5])*fact
        result[1] = add.reduce(a4[1,:]*signal[:5])*fact
        gj = zeros((len(result)-4,5),'d')
        gj[:,0]   = a4[2,0]*signal[:-4]
        gj[:,1]   = a4[2,1]*signal[1:-3]
        gj[:,2]   = a4[2,2]*signal[2:-2]
        gj[:,3]   = a4[2,3]*signal[3:-1]
        gj[:,4]   = a4[2,4]*signal[4:]
        result[2:-2] = add.reduce(gj,-1)*fact
        result[-2] = add.reduce(a4[3,:]*signal[-5:])*fact
        result[-1] = add.reduce(a4[4,:]*signal[-5:])*fact
    elif diffScheme == 'order 5':
        fact = 1./(120.*dt)
        result = zeros(len(signal),'d')
        result[0] = add.reduce(a5[0,:]*signal[:6])*fact
        result[1] = add.reduce(a5[1,:]*signal[:6])*fact
        result[2] = add.reduce(a5[2,:]*signal[:6])*fact
        gj = zeros((len(result)-5,6),'d')
        gj[:,0]   = a5[3,0]*signal[:-5]
        gj[:,1]   = a5[3,1]*signal[1:-4]
        gj[:,2]   = a5[3,2]*signal[2:-3]
        gj[:,3]   = a5[3,3]*signal[3:-2]
        gj[:,4]   = a5[3,4]*signal[4:-1]
        gj[:,5]   = a5[3,5]*signal[5:]
        result[3:-2] = add.reduce(gj,-1)*fact
        result[-2] = add.reduce(a5[4,:]*signal[-6:])*fact
        result[-1] = add.reduce(a5[5,:]*signal[-6:])*fact
    else: result = None
    return result        
        

def rotMatrix(q):
    """ rotation matrix from quaternions """
    D = zeros((len(q),3,3),Float)
    
    D[:,0,0] = -2.*q[:,2]**2 - 2.*q[:,3]**2 + 1.
    D[:,0,1] = 2.*(-q[:,0]*q[:,3] + q[:,1]*q[:,2])
    D[:,0,2] = 2.*(q[:,0]*q[:,2] + q[:,1]*q[:,3])

    D[:,1,0] = 2.*(q[:,0]*q[:,3] + q[:,1]*q[:,2])
    D[:,1,1] = -2.*q[:,1]**2 - 2.*q[:,3]**2 + 1.
    D[:,1,2] = 2.*(-q[:,0]*q[:,1] + q[:,2]*q[:,3])

    D[:,2,0] = 2.*(-q[:,0]*q[:,2] + q[:,1]*q[:,3])
    D[:,2,1] = 2.*(q[:,0]*q[:,1] + q[:,2]*q[:,3])
    D[:,2,2] = -2.*q[:,1]**2 - 2.*q[:,2]**2 + 1.

    return D


## def WignerFunctions(j,m,n,q):
    
##     pmin = max(0,m-n)
##     pmax = min(j+m,j-n)
##     a = q[:,0] + 1j*q[:,3]
##     b = q[:,2] + 1j*q[:,1]
##     a_star = conjugate(a)
##     b_star = conjugate(b)

##     p = pmin
##     D = 0.0+0j
##     while p<=pmax:
##         D = D + (-1)**(p+n-m) * sqrt(factorial(j+m)*factorial(j-m)*\
##                                      factorial(j+n)*factorial(j-n)) *\
##             a**(j+m-p) * a_star**(j-n-p) * b**(p+n-m) * b_star**p /\
##             (factorial(j+m-p)*factorial(j-n-p)*factorial(p)*factorial(p+n-m))
##         p = p+1

##     return D

def WignerFunctions(j, m, n, q):
    t = max(0, m-n)
    try:
        sign, coeff, den1, den2, den3, den4 = _w_coeff_cache[(j, m, n)]
    except KeyError:
        factors = range(j+m) + range(j-m) + range(j+n) + range(j-n)
        coeff = sqrt(multiply.reduce(1.+array(factors, Float)))
        factors = range(j+m-t) + range(j-n-t) + range(t) + range(t+n-m)
        coeff = coeff/multiply.reduce(1.+array(factors, Float))
        sign = 1.-2*(t % 2)
        den1 = float(j+m-t)
        den2 = float(j-n-t)
        den3 = float(t)
        den4 = float(t+n-m)
        _w_coeff_cache[(j, m, n)] = sign, coeff, den1, den2, den3, den4

    a = q[:, 0]+1j*q[:, 3]
    b = q[:, 2]+1j*q[:, 1]
    a_conj = conjugate(a)
    b_conj = conjugate(b)

    f1 = a**(j+m-t)
    f2 = a_conj**(j-n-t)
    f3 = b**(t+n-m)
    f4 = b_conj**(t)
    sum = sign*coeff*f1*f2*f3*f4
    for t in range(max(0, m-n), min(j+m, j-n)):
        sign = -sign
        f1 = f1/a
        f2 = f2/a_conj
        f3 = f3*b
        f4 = f4*b_conj
        den3 = den3+1
        den4 = den4+1
        coeff = coeff*den1*den2/(den3*den4)
        den1 = den1-1
        den2 = den2-1
        sum = sum + sign*coeff*f1*f2*f3*f4
    return sum

_w_coeff_cache = {}


def histogram(data,bins):

    n = searchsorted(sort(data),bins)
    n = concatenate([n,[len(data)]])
    return n[1:] - n[:-1]


class qTrajectory(Trajectory):

    def __init__(self, object, filename, mode = 'r', comment = None,
                 double_precision = 0, cycle = 0):

        Trajectory.__init__(self, object, filename, mode = 'r', comment = None,
                            double_precision = 0, cycle = 0)
        group_desc = eval(self.trajectory.file.variables['group_description']\
                        [:].tostring())
        self.atoms = {}
        self.groups = {}
        for group in range(len(group_desc)):
            for atom in group_desc[group]:
                self.atoms[atom] = group
            self.groups[group_desc[group][0]] = group

    def readParticleTrajectory(self, atom, first=0, last=None, skip=1,
                               variable = "configuration"):
        """ przypisujemy atom do danej grupy, znajdujemy index grupy,
        czytamy cms dla tej grupy, zwracamy x,y,z:
        w ten sposob DOS, VAF itp beda wagowane przez mase grupy
        (kazdy atom w grupie (znamy jego mase) bedzie mial ta sama
        trajektorie) """
        if last is None: last = len(self)
        grindex = self.atoms[atom.index]
        gj = Quidam()
        gj.array = self.trajectory.file.variables['cms']\
                   [first:last:skip,grindex].astype(Float)
        return gj

    def readRigidBodyTrajectory(self, object, first=0, last=None, skip=1,
                                reference = None):
        """ Zwroc obiekt ktory ma atrybuty cms, quaternions i fit,
        Ktora grupe czytac (index bedziemy wiedziec po pierwszym atomie
        w jej atomList() """
        if last is None: last = len(self)
        grindex = self.groups[object.atomList()[0].index]
        gj = Quidam()
        gj.cms = self.trajectory.file.variables['cms']\
                 [first:last:skip,grindex].astype(Float)
        gj.quaternions = self.trajectory.file.variables['quaternion']\
                         [first:last:skip,grindex].astype(Float)
        gj.fit = self.trajectory.file.variables['fit']\
                 [first:last:skip,grindex].astype(Float)
        return gj


class Quidam:

    def __init__(self):

        pass

    def __getattr__(self, attr):
        return None
