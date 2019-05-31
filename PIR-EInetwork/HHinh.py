"""
Template for single compartment HH model of cortical interneuron

Coded: Ruben A. Tikidji-Hamburyan

"""
import os,sys,csv
from numpy import *
from numpy import random as rnd
import scipy as sp
import matplotlib
matplotlib.rcParams["savefig.directory"] = ""
from matplotlib.pyplot import *
try:
	import cPickle as pkl
except:
	import pickle as pkl
from neuron import h

class In:
	def __init__(self, init=-64.9736783327):
		self.soma = h.Section()
		self.soma.L		= 20.
		self.soma.diam	= 2./np.pi
		self.soma.nseg	= 1
		self.soma.insert('hh')
		#self.soma.gnabar_hh = 1.2
		#self.soma.gkbar_hh = 0.36
		#self.soma.gl_hh = 0.003/2.
		#self.soma.gnabar_hh *= 2.8
		#self.soma.gkbar_hh *= 2.8
		#self.soma.gl_hh *= 2.8

		self.soma.v = init
		#Spike Recorder
		self.spks	= h.Vector()
		self.recorder = h.NetCon(self.soma(0.5)._ref_v,None,sec=self.soma)
		self.recorder.threshold = 0.
		self.recorder.record(self.spks)
		#Synapses
		self.isyn	= h.Exp2Syn(0.5, sec=self.soma)
		self.isyn.e		= -75.0
		self.isyn.tau1	= 1.
		self.isyn.tau2	= 3.0
		self.esyn	= h.Exp2Syn(0.5, sec=self.soma)
		self.esyn.e		= 0.0
		self.esyn.tau1	= 0.8
		self.esyn.tau2	= 1.2
		#Recorders
		self.rec	= {}
		#output connection point
		self.output  = self.soma(0.5)._ref_v

	def setrecorder(self,name):
		if type(name) is int or type(name) is float or type(name) is list or type(name) is tuple or type(name) is str:
			raise TypeError("first argument should be only dict")
		for n in name:
			vec = h.Vector()
			try:
				exec "vec.record(self."+name[n]+")"
			except:
				print "Cannot set a recorder for self."+name[n]
			self.rec[n]  = vec
	def getEtoIspace(self,conduct=False):
		if conduct: 
			return (self.esyn.tau2 - self.esyn.tau1)/(self.isyn.tau2 - self.isyn.tau1)
		else:
			#return (self.esyn.tau2 - self.esyn.tau1)*abs(self.esyn.e-self.soma(0.5).v)/(self.isyn.tau2 - self.isyn.tau1)/abs(self.isyn.e-self.soma(0.5).v)
			return (self.esyn.tau2 - self.esyn.tau1)*abs(self.esyn.e+64.9736783327)/(self.isyn.tau2 - self.isyn.tau1)/abs(self.isyn.e+64.9736783327)
		
		
if __name__ == "__main__":

	#h.celsius = 20.
	iin = In()
	iin.setrecorder({"volt":"soma(0.5)._ref_v"})
	
	stim = h.IClamp(0.5,sec=iin.soma)
	stim.amp = 0.003
	stim.delay = 100.
	stim.dur   = 300.
	
	trec = h.Vector()
	trec.record(h._ref_t)
    
	h.finitialize()
	h.fcurrent()
	h.frecord_init()
	while h.t < 500. :h.fadvance()
    
    
	plot(trec,iin.rec["volt"])
	c = array(iin.spks)
	print float(c.shape[0])/300.*1000, iin.soma(0.5).v
	show()
