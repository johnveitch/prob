#import FuncDesigner
#from FuncDesigner import *
from __future__ import division
import scipy
import scipy.integrate
from numpy import inf,linspace,exp,log,logaddexp
import numpy as np
import types
import functools

integrate=scipy.integrate.quad

class Reals(object):
    # Class representing real numbers
    def __init__(self,min=-inf,max=inf):
        self.min=min
        self.max=max
    def __str__(self):
        return 'Real in (%s,%s)'%(str(self.min),str(self.max))
    def __repr__(self):
        return 'Real(min=%s,max=%s)'%(str(self.min),str(self.max))
    def discretise(self,N=100):
        return linspace(self.min,self.max,N)

class Integers(object):
    # Class representing integers
    def __init__(self,min=-inf,max=inf):
        self.min=min
        self.max=max
    def __str__(self):
        return 'Integer in (%s,%s)'%(str(self.min),str(self.max))
    def __repr__(self):
        return 'Integers(min=%s,max=%s'%(str(self.min),str(self.max))
    def discretise(self):
        return range(self.min,self.max)

class Distribution(object):
    # A probability distribution parent class, only defines interface
    def __init__(self,*args,**kwargs):
        pass
    def derivative(self, *args, **kwargs):
        # Returns derivative w.r.t params listed in args
        # i.e. f.derivative(['x']) = df/dx
        pass
    def marginalise(self, *args, **kwargs):
        # Marginalise this distribution over the domain specified in **kwargs
        # e.g. f.marginalise({'x':Reals(0,10)}) = Int_0^10 f(x) dx
        pass
    def bind(self,**kwargs):
        # Return the curried function with argument bound, returns a function
        # e.g. f.bind(x=10) = lambda : f(10)
        pass
    def __call__(self,**kwargs):
        # Evaluate the function, returns a real number
        # e.g. f(x=10) = f(10)
        pass
    def __add__(self,other):
        # Creates the summed distribution f+g = f(x)+g(x)
        pass
    def __mul__(self,other):
        # Creates the product of distributions f*g = f(x)*g(x)
        pass

class LogDistributionFunction(Distribution):
    # Class representing a analytic distribution function
    def __init__(self,func,**kwargs):
        self.vars=kwargs.keys()
        self.func = func
        self.domain = kwargs
        self.factors=[]
    def __str__(self):
        return 'Distribution %s on domain %s'%(str(self.func),str(self.domain))
    def pick_args(self,**kwargs):
        return {k[0]:k[1] for k in kwargs.items() if k[0] in self.vars}
    def bind(self,**kwargs):
        f = functools.partial(self.func,**kwargs)
        keys=kwargs.keys()
        dom = {k:v for k,v in self.domain.iteritems() if k not in keys }
        return LogDistributionFunction(f,**dom)
    def __call__(self,**kwargs):
        bounddist=self.bind(**kwargs)
        if len(bounddist.vars)==0: return bounddist.func()
        else: return bounddist
    def expcall(self,**kwargs):
        bounddist=self.bind(**kwargs)
        if len(bounddist.vars)==0: return exp(bounddist.func())
        else: return bounddist
    def marginalise(self,**kwargs):
        if len(kwargs)==0:
            return self
        if len(self.factors)!=0:
            var,domain = kwargs.popitem()
            samefacs=[]
            for factor in self.factors:
                if var in factor.vars:
                    newfac=factor.marginalise(**{var:domain})
                else:
                    samefacs.append(factor)
            newdist=newfac
            for d in samefacs:
                newdist=newdist*d
            return newdist.marginalise(**kwargs)
        if len(kwargs)==1:
            var,domain = kwargs.popitem()
            subdomain=self.domain.copy()
            if var in subdomain.keys(): subdomain.pop(var)
            if isinstance(domain,Reals):
                f=lambda **other: log(integrate(lambda a:self.bind(**other).expcall(**{var:a}),domain.min,domain.max)[0])
            elif isinstance(domain,Integers):
                f=lambda **other: reduce(logaddexp,map(lambda a:self.bind(**other).expcall(**{var:a}),domain))\
                        -log(domain.max-domain.min)
            return LogDistributionFunction(f,**subdomain)
        else:
            var,domain = kwargs.popitem()            
            subspace=self.marginalise(**{var:domain})
            return subspace.marginalise(**kwargs)
    def MCintegrate(self,samples,colnames):
        """
        Dist.MCintegrate(samps, dict([('a',0),('b',1)])) = 1/N * sum_i Dist(a_i,b_i)

        colnames: a dictionary of (name,index) where name is the name of the variables
        you want to marginalise out, and index is the column of that in the samples matrix.
        """
        N=len(samples)
        samples=scipy.array(samples,ndmin=2)
        otherdomain={key:val for (key,val) in self.domain.iteritems() if key not in colnames.keys()}
        def getargs(row,colnames):
            return {name:row[i] for (name,i) in colnames.iteritems()}
        def newfunc(**other):
                return reduce(logaddexp, \
                  [self.bind(**getargs(row,colnames))(**other) for row in samples]) \
                -log(N)
        return LogDistributionFunction(newfunc,**otherdomain)

    def __mul__(self,other):
        """
        p(x,y) = p(x)*p(y)
        p(x,y,z) = p(x,y)*p(x,z)
        """
        seperable = True
        if isinstance(other,Distribution):
            dom1=self.domain
            dom2=other.domain
            outerdom = dict(dom1.items()+dom2.items())
            for k in outerdom.keys():
                if dom1.has_key(k) and dom2.has_key(k):
                    seperable = False
                    if dom2[k].min != dom1[k].min or dom2[k].max != dom1[k].max:
                       raise Exception('Cannot take outer product of two functions of same variable %s with different domains'%(k))
            if cmp(dom1,dom2)==0:
                f=lambda **dom:self.func(**dom)+other.func(**dom)
            else:
                f=lambda **dom: self.func(**self.pick_args(**dom)) + other.func(**other.pick_args(**dom))
            dist= LogDistributionFunction(f,**outerdom)
            if seperable:
                dist.factors=[self,other]
            return dist
        if isinstance(other,float) or isinstance(other,int):
            outerdomain=self.domain
            f=lambda **outerdomain:log(other)+self.func(**outerdomain)
            return LogDistributionFunction(f,**self.domain)
        else:
            raise Exception('Cannot take product of %s and %s'%(type(self),type(other)))

    def __add__(self,other):
        if not isinstance(other,Distribution):
            raise Exception('Cannot add distribution and %s'%(type(other)))
        if cmp(self.domain,other.domain) != 0:
            raise Exception('Cannot sum two functions with different domains %s and %s'%(self.domain,other.domain))
        def f(**dom):
            #print 'lvar=%s, rval=%s'%(self.func(**dom),other.func(**dom))
            return logaddexp(self.func(**dom),other.func(**dom))
        return LogDistributionFunction(f,**self.domain)
    def __pow__(self,x):
        return LogDistributionFunction(lambda **dom:self.func(**dom)*x,**self.domain)

def makeProductDistribution(distributions):
    """
    Make the distribution that is the product of the list of distributions given
    so that it can be evaluated efficiently
    """
    def newfunc(**dom):
        return sum( [d.func(**dom) for d in distributions] )
    newdom = {}
    for d in distributions:
        newdom.update(d.domain)
    return LogDistributionFunction(newfunc,**newdom)


class tests(object):
   # Unit tests 
   def __init__(self):
       self.passed=True
       self.creation()
       self.MCintegrate()
   def creation(self):
       import math
       from math import pi,exp,sqrt
       self.x=Reals()
       print 'Created real x: %s'%(str(self.x))
       self.F=LogDistributionFunction(lambda x:-0.5*x**2 - 0.5*log(2*pi), x=self.x )
       print 'Created distribution %s'%(self.F)
       print 'F(0)=%s'%(str(self.F.expcall(x=0)))
       print 'Marginalised normalised Gaussian, result = %s'%(self.F.marginalise(x=self.x).expcall())
   def MCintegrate(self):
        print 'Testing Monte Carlo integration on normalised Gaussian'
        samps = (np.random.randn(40000),np.random.random(40000))

        #d=DistributionSamples(samps,['a','b'])
        avar=Reals(min=-100,max=100)
        bvar=Reals(min=-100,max=100)
        cvar=Reals()

        print 'Testing 1D'

        g=LogDistributionFunction(lambda a:-0.5*a**2 -0.5*log(2.0*np.pi),a=Reals())
        gdash = LogDistributionFunction(lambda a:log(1.0),a=Reals())
        colnames=dict([('a',0)])
        result = gdash.MCintegrate(samps,colnames).expcall()
        marg=(g*gdash).marginalise(a=avar).expcall()
        print 'numerical: %f, monte carlo: %f'%(marg,result)

        print 'Testing 2D'

        g=LogDistributionFunction(lambda a,b:-0.5*(a**2 + b**2 ) -0.5*2.0*log(2.0*np.pi) ,a=Reals(),b=Reals())
        gdash=LogDistributionFunction(lambda a,b:log(1.0),a=Reals(),b=Reals())
        colnames=dict([('a',0),('b',1)])
        result=gdash.MCintegrate(samps,colnames).expcall()
        marg=(g*gdash).marginalise(a=avar,b=bvar).expcall()
        print 'numerical: %f, monte carlo: %f'%(marg,result)

        print 'Testing 1D monte carlo, 1D numerical'
        colnames=dict([('a',0)])
        Bdist=LogDistributionFunction(lambda b:-0.5*b**2 - 0.5*log(2.0*np.pi),b=Reals())
        Adist=LogDistributionFunction(lambda a:-0.5*a**2 - 0.5*log(2.0*np.pi),a=Reals())

        result=((Bdist*gdash).MCintegrate(samps,colnames)).marginalise(b=bvar).expcall()
        print 'MC then numerical: %f'%(result)
        result=((Bdist*gdash).marginalise(b=bvar)).MCintegrate(samps,colnames).expcall()
        print 'numerical then MC: %f'%(result)


if __name__=='__main__':
    t=tests()
    
