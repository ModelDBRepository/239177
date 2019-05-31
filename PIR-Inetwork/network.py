# -*- coding: utf-8 -*-
"""
/***********************************************************************************************************\

 This NEURON + Python scripts associated with paper:                                                        
  Ruben A. Tikidji-Hamburyan, Carmen C. Canavier
  Robust One and Two Cluster Synchrony Mediated by Resonant Interneurons in Sparsely but 
    Strongly Connected Inhibitory and Excitatory/Inhibitory Networks
                             
                                                                                                           
 Network of 300 H-H type-II neurons connected by double-exponential synapses is modeled                
 All parameters may be set up by command line arguments listed AFTER script name. The command should be:   
  nrngui -nogui -python network.py [PARAMETERS]                                                            
 A list of avalible parameters can be printed out by command:                                              
  nrngui -nogui -python network.py --help
                                                                                                           
 ---
 
 To replicate activity for any parameter set in bifurcation diagram Figure 2 A and B run:                                                                                
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /ncon=\\(\\\'b\\\',0.133\) /synapse/weight=<synaptic conductance in uS> /synapse/delay=<delay in milliseconds>
  
 For example:
 for 2 clusters inset:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /ncon=\\(\\\'b\\\',0.133\) /synapse/weight=0.006e-2 /synapse/delay=0.8
 for synchrony inset:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /ncon=\\(\\\'b\\\',0.133\) /synapse/weight=0.1e-2 /synapse/delay=3.
 for sparce synchrony inset:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /ncon=\\(\\\'b\\\',0.133\) /synapse/weight=0.1e-2 /synapse/delay=0.8

 ---
 
 To replicate activity for any parameter set in bifurcation diagram Figure 3A run:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /ncon=\\(299,299,40\\) /synapse/weight=<synaptic conductance in uS> /synapse/delay=<delay in milliseconds>

 For example:
 for 2 clusters inset:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /ncon=\\(299,299,40\\) /synapse/weight=0.006e-2 /synapse/delay=0.8
 for synchrony inset:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /ncon=\\(299,299,40\\) /synapse/weight=0.1e-2 /synapse/delay=3.
 
 ---
 
 To replicate activity for noise current in Figure 4 A and B run:
 for 2 cluster mode:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.006e-2 /synapse/delay=0.8 /neuron/Istdev=0.16e-2*<required noise CV>
 for synchrony :
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2 /synapse/delay=3. /neuron/Istdev=0.16e-2*<required noise CV>
 One can check spontaneous firing rate for specific noise by setting connection weight to zero: 
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0  /neuron/Istdev=0.16e-2*<required noise CV>
  
 For the same simulations to see a difference between noise-less and noise-full network as in Figures 4 C and D, run:
 for 2 cluster mode:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.006e-2 /synapse/delay=0.8 /nstart/Istdev=0.16e-2*<required noise CV>  /nstart/delay=250
 for Figure 4C:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.006e-2 /synapse/delay=0.8 /nstart/Istdev=0.16e-2*3  /nstart/delay=250
 for synchrony : 
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2 /synapse/delay=3. /nstart/Istdev=0.16e-2*<required noise CV>  /nstart/delay=250
 for Figure 4D:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2 /synapse/delay=3. /nstart/Istdev=0.16e-2*3  /nstart/delay=250
 
 ---
                                                                                                       
 To replicate points from Figure 5 A and B run:
 for 2 cluster mode:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.006e-2 /synapse/delay=0.8 /synapse/gsynscale=\(1,<required CV in total synaptic weight>\) /gtot-dist=\\\'LOGN\\\'
 for synchrony : 
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2 /synapse/delay=3. /synapse/gsynscale=\(1,<required CV in total synaptic weight>\) /gtot-dist=\\\'LOGN\\\'
 
 To replicate examples:
 in Figure 5 C:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.006e-2 /synapse/delay=0.8 /gtot-dist=LOGN /synapse/gsynscale=\(1.,1.5\) /Gtot-dist=True /sortbysk=GT 
 in Figure 5 D:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2 /synapse/delay=3. /gtot-dist=LOGN /synapse/gsynscale=\(1.,1.5\) /Gtot-dist=True /sortbysk=GT 
                                                                                                           
 ---

 To replicate points from Figure 5 A and B run:
 for 2 cluster mode:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.006e-2 /synapse/delay=\(0.3,0.8*<required delay CV>,0.5\) /Delay-stat=True /delay-dist=LOGN-SHIFT /tv=\(0,1000\)
 for synchrony : 
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2   /synapse/delay=\(3,3.*<required delay CV>,0.5\)    /Delay-stat=True /delay-dist=LOGN-SHIFT /tv=\(0,1000\) 
 for synchrony with truncation at 1.2 ms: 
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2   /synapse/delay=\(3,3.*<required delay CV>,1.2\)    /Delay-stat=True /delay-dist=LOGN-SHIFT /tv=\(0,1000\) 

 To replicate examples:
 in Figure 6 C:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.006e-2 /synapse/delay=\(0.3,0.8*3,0.5\) /Delay-stat=True /delay-dist=LOGN-SHIFT /tv=\(0,1000\)
 in Figure 6 D:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2   /synapse/delay=\(1.8,3.*3,1.2\) /Delay-stat=True /delay-dist=LOGN-SHIFT /tv=\(0,1000\) 
 in Figure 6 E:
  nrngui -nogui -python network.py /gui=True /preview=True /git=False /synapse/weight=0.1e-2   /synapse/delay=\(3,3.*3,0.5\)    /Delay-stat=True /delay-dist=LOGN-SHIFT /tv=\(0,1000\) 


 Copyright: Ruben Tikidji-Hamburyan <rath@gwu.edu> <rtikid@lsuhsc.edu> Dec.2015 - Mar.2017

\************************************************************************************************************/
"""
import numpy as np
import scipy.stats as sps
import sys,os,csv,threading
import random as rnd
try:
	import cPickle as pickle
except:
	import pickle
from datetime import datetime
import time

from neuron import h
h.load_file("stdgui.hoc")

###### Abbreviations:
Abbreviations=(
	( 'I',       'I',   "Current"),
	( 'TI',      'TI',  "Total Current"),
	( 'TSI',     'TSI', "Total Synaptic Current"),
	( 'MTI',     'MTI', "Mean Total Curent"),
	( 'MTSI',    'MTSI',"Mean Total Synaptic Current"),
	( 'G',       'G',   "Condunctance"),
	( 'TG',      'TG',  "Total Conductance"),
	( 'MTG',     'MTG', "Mean Total Conducntance"),
	( 'FR',      'FR',  "Firing Rate"),
	( 'NORM',   'NORM',"Normal distribution"),
	( 'LOGN',   'LOGN',"Log Normal Distribution"),
	( 'Mstate', 'm',   "Sodium Activation variable"), 
	( 'Hstate', 'h',   "Sodium Inctivation variable"),
	( 'Nstate', 'n',   "Potassium Activation variable"),
	( 'ucon',   'ucon',"Uniform distribution of number connection per cell"),
	( 'ncon',   'ncon',"Normal distribution of number connection per cell"),
	( 'bcon',   'bcon',"Binomial distribution of number connection per cell"),
)

for ab,val,meaning in Abbreviations:
	print "Applay abbreviation % 8s for % 8s for: "%(ab,val)+meaning,
	try:
		exec ab+'=\''+val+'\''
	except:
		print "Fail!"
		exit(1)
	print "DONE"

###### Paramters:
methods		= {
	'ncell'		: 300,		# number of neurons in population
	'ncon'		: 40,		# number of input connections per neuron
							# constant or uniform distribution(from, to) or normalized uniform distribution (mena, stder, ncon-norm)
							# OR uniform distribution  ('u', from, {to},    {{ncon-norm}})
							# OR normal distribution   ('n', mean, {stdev}, {{ncon-norm}})
							# OR binomial distribution ('b', prob,           {ncon-norm} )
	'neuron'	: {
		'Vinit'		: (-50.,20),#(-51.86007190636312,20),	# Constant or (mean,stdev) or [value for each neuron] or string or file name there values for each neuron are contained.
		'Iapp'		: 0.16e-2,#None,				# Same or None 
		'Istdev'	: None,						# ---/---/---
	},
	'synapse'	: {
		'weight'	: 0.006e-2,					# Synaptic conductance
		'delay'		: 0.8,						# Axonal Delays
		'gsynscale'	: 1.0,						# Conductance caramelization
		'tau1'		: 1.0,
		'tau2'		: 3.0,
		'Esyn'		: -75.0,					# Synaptic reversal potential
												# Constant or (mean,stdev) or [value for each neuron] or string or file name there values for each neuron are contained.
		'synscaler'	: None,

	},		
	'R2'		: True,
	'maxFreq'	: 200.0,		# max frequency
	'peakDetec' : True,			# Turn on/off peak detector
	'gkernel'	: (5.0,25.0),#(10.0,50.0),	# Kernel and size (5,25),#
	"netFFT"	: False,#True,#False,#True,#False,		# Turn on/off network FFT
	"nrnFFT"	: False,#True,#False,		# Turn on/off neuron FFT
	'netISI'	: 30001,			# max net ISI
	'nrnISI'	: 30001,			# max neuron ISI
	'cliptrn'	: False,#1000,#False alse,#500,#False,	# Clip transience for first n ms or False
	'traceView'	: False,#'n',
	'tV-synmax' : False,
	'traceWidth': 55.0,			#
	'tracetail'	: 'mean total conductance',#'conductance',#'current',#'conductance',#'firing rate',#'total conductance', #'total current' 'total current',
	'patview'	: True,			# Turn on/off Pattern view
	'gui'		: True,
	'git'		: False,		# Turn on/off git core (Never turn-on at the head node!!!!!!!)
	'gif'		: False,		# Generate gif instead pop up on a screan.
	'corefunc'	: (4,8,64),
	'coreindex'	: False,			# Turn on/off Core indexing
	'corelog'	: 'network',
	'noexit'	: False,
	'GPcurve'	: False,
	'IGcurve'	: False,
	'Conn-rec'	: False,
	'Conn-stat'	: False,
	'G-Spikerate'	: False,
	
	'Gtot-dist'	: False,
	'Gtot-rec'	: False,#True,	#record all gtotal in neurons
	'Gtot-stat'	: True, #False, #record gtot statistic
	'sycleprop'	: False,#True,
	'external'	: False,
	'extprop'	: 0.5,				# Calculate probability to fire after external input
	'timestep'	: 0.01,#0.005,
	'sortbysk'	: False, #'ST',#'F',#'GT',#'NC',#'GT','G',#'I', #'F',#'I',#'F',#False,			#Do not use
	'taunorm'	: False,#True,#False,#True,
	'nstart'	: False, #(900.,0.1e-5,1000),<Noise for paper  #False,#(900.,0.2e-5,1000),#False,#(200,0.000002,900),#False, (delay, ampl, dur)
	'cliprst'	: False,#10,#False,#20,
	'T&S'		: True,
	'lastspktrg': True,
	'fullrast'	: True,
	'gtot-dist'	: 'NORM',#'LOGN', #LOGN - lognormal, 'NORM' - normal
	'gsyn-dist' : 'LOGN',#'NORM',#'LOGN', #same
	'cycling'	: False, #4,False
	'popfr'		: False,#True,	#calculate population firing rate
	'cmd-file'	: 'network.start',
	'preview'	: True,
	'2cintercon': False,#True,
	'2clrs-stat': False,#True,
	'tv'		: (0., 500.),
	'tstop'		: 10001,
	'jitter-rec': False,#True,
}



class neuron:
	def __init__(self):
		self.soma = h.Section()
		if checkinmethods('/neuron/L'):
			self.soma.L		= methods["neuron"]["L"]
		else:
			self.soma.L		= 20.
		if checkinmethods('/neuron/diam'):
			self.soma.diam	= methods["neuron"]["diam"]
		else:
			self.soma.diam	= 2./np.pi
		if checkinmethods('/neuron/nseg'):
			self.soma.nseg	= int(methods["neuron"]["nseg"])
		else:
			self.soma.nseg	= 1
		if checkinmethods('/neuron/cm'):
			self.soma.cm	= float(methods["neuron"]["cm"])
		self.soma.insert('hh')
		#self.soma.insert('hhlim')
		self.soma(0.5).v = -63.
		self.isyn	= h.Exp2Syn(0.5, sec=self.soma)
		self.isyn.e		= -75.0
		self.isyn.tau1	= 2.0
		self.isyn.tau2	= 10.0
		######## Recorders ##########
		self.spks	= h.Vector()
		self.sptr	= h.APCount(.5, sec=self.soma)
		self.sptr.thresh = 0.
		self.sptr.record(self.spks)
		#self.sptr = h.NetCon(self.soma(0.5)._ref_v,None,sec=self.soma)
		#self.sptr.threshold = 25.
		#self.sptr.record(self.spks)
		if checkinmethods('gui'):
			self.volt	= h.Vector()
			self.volt.record(self.soma(0.5)._ref_v)
			if checkinmethods("neuron/record/current"):
				self.isyni	= h.Vector()
				self.isyni.record(self.isyn._ref_i)
			if checkinmethods('neuron/record/conductance'):
				self.isyng	= h.Vector()
				self.isyng.record(self.isyn._ref_g)
			if checkinmethods('traceView'):
				self.svar   = h.Vector()
				exec "self.svar.record(self.soma(0.5)._ref_"+methods['traceView']+"_hh)"
		elif checkinmethods('get-steadystate'):
			self.volt	= h.Vector()
			self.volt.record(self.soma(0.5)._ref_v)
		if checkinmethods('sinmod'):
			self.sin = h.sinIstim(0.5, sec=self.soma)
			if type(methods['sinmod']) is dict:
				for name in methods['sinmod']:
					exec "self.sin."+name+"= {}".format(methods['sinmod'][name])

		######## Registrations ###### 
		self.gsynscale	= 0.0
		self.concnt		= 0.0
		self.gtotal		= 0.0
		self.tsynscale	= 1.0

	def setparams(self, 
			V=None, 
			Iapp = None, Insd = None, delay = None, duration = None, period = None,
			SynE = None, SynT1 = None, SynT2 = None,
			):
		if not V is None : self.soma(0.5).v = V
		########
		if not Iapp is None or not Insd is None:
			self.innp	= h.InNp(0.5, sec=self.soma)
			self.rnd	= h.Random(np.random.randint(0,32562))
			self.innp.noiseFromRandom(self.rnd)
			self.innp.dur	= 1e9 if duration == None else duration
			self.innp.delay	= 0   if delay == None else delay
			self.innp.per	= 0.1 if period == None else period
			self.innp.mean	= 0.0 if Iapp == None else Iapp
			self.innp.stdev	= 0.0 if Insd == None else Insd
			if methods['gui']:
				self.inoise	= h.Vector()
				self.inoise.record(self.innp._ref_i)
			elif checkinmethods("rawdata") and type(methods["rawdata"]) is str:
				self.inoise	= h.Vector()
				self.inoise.record(self.innp._ref_i)
		########
		if not SynE  is None: self.isyn.e		= SynE
		if not SynT1 is None: self.isyn.tau1	= SynT1
		if not SynT2 is None: self.isyn.tau2	= SynT2
		########
			
			

	def addnoise(self,Iapp=0.,Insd=0.,delay=0.,dur=0.,per=0.1):
		self.andnoise = h.InNp(0.5, sec=self.soma)
		self.andrnd	= h.Random(np.random.randint(0,32562))
		self.andnoise.noiseFromRandom(self.andrnd)
		self.andnoise.mean  = Iapp
		self.andnoise.stdev = Insd
		self.andnoise.delay	= delay
		self.andnoise.per	= per
		self.andnoise.dur	= dur

#class symulation:
	#def __init___(self,params):
		#if params.get("a",False):

def onclick1(event):
	if not hasattr(onclick1,"aix"):
		onclick1.aix=zooly.add_subplot(111)
	onclick1.et = event.xdata
	
	### BUG
	onclick1.tl, onclick1.tr = onclick1.et-methods['traceWidth'], onclick1.et+methods['traceWidth']
	onclick1.idx, = np.where( (t > onclick1.tl) * (t < onclick1.tr))
	
	if not hasattr(onclick1,"marks"):
		onclick1.marks = []
		onclick1.marks.append( p.plot([onclick1.tl,onclick1.tl],[-80,30],"r--",lw=2)[0] )
		onclick1.marks.append( p.plot([onclick1.tr,onclick1.tr],[-80,30],"r--",lw=2)[0] )
	else:
		onclick1.marks[0].set_xdata([onclick1.tl,onclick1.tl])
		onclick1.marks[1].set_xdata([onclick1.tr,onclick1.tr])
	
	if not hasattr(onclick1,"lines"):
		onclick1.lines = []
		for n in neurons:
			volt = np.array(n.volt)
			onclick1.lines.append(onclick1.aix.plot(t[onclick1.idx],volt[onclick1.idx])[0])
	else:
		vmin,vmax = 1000,-1000
		for ind,n in map(None,xrange(methods["ncell"]),neurons):
			volt = np.array(n.volt)
			if vmin > volt[onclick1.idx].min():vmin = volt[onclick1.idx].min()
			if vmax < volt[onclick1.idx].max():vmax = volt[onclick1.idx].max()
			onclick1.lines[ind].set_xdata(t[onclick1.idx])
			onclick1.lines[ind].set_ydata(volt[onclick1.idx])
			onclick1.lines[ind].set_linewidth(1)
			onclick1.lines[ind].set_ls("-")

		onclick1.aix.set_xlim(onclick1.tl,onclick1.tr)
		#print vmin,"---",vmax
		onclick1.aix.set_ylim(vmin,vmax)
	if hasattr(moddyupdate,"lines"):
		del moddyupdate.lines
	mainfig.canvas.draw()
	zoolyupdate(vindex)
	moddyupdate(moddyupdate.idx)



def getnulls(vmin,vmax,gsyn,inoise,ibias):
	vtrap = np.vectorize( lambda x,y: y*(1 - x/y/2) if np.abs(x/y) < 1e-6 else x/(np.exp(x/y) - 1) )
	gnabar = neurons[0].soma(0.5).hh.gnabar
	gkbar = neurons[0].soma(0.5).hh.gkbar
	ena,ek = neurons[0].soma.ena, neurons[0].soma.ek
	gl,el = neurons[0].soma(0.5).hh.gl, neurons[0].soma(0.5).hh.el
	s = neurons[0].soma.L * neurons[0].soma.diam * np.pi
	#es = methods["synapse"]["Esyn"]
	es = neurons[0].isyn.e
	#DB>>
	#print gnabar,gkbar,gl,s,gsyn,inoise,ibias
	#<<DB
	## N -null
	vx=np.linspace(vmin,vmax,200)
	nalph,nbeta =.01*vtrap(-(vx+55.),10.), .125*np.exp(-(vx+65.)/80.)
	u0=nalph/(nalph + nbeta)
	## V - null
	malph,mbeta =.1*vtrap(-(vx+40.),10), 4.*np.exp(-(vx+65.)/18.)
	m = malph / (malph + mbeta)
	halph,hbeta =.07 * np.exp(-(vx+65.)/20.), 1. / (np.exp(-(vx+35.)/10.) + 1)
	h = halph / (halph + hbeta)
	v0 = np.sqrt( np.sqrt( ( -1.e-5*gnabar*m*m*m*h*(vx-ena) - 1.e-5*gl*(vx- el)-ibias*1e-3/s ) /
		( 1.e-5 * gkbar * (vx- ek) ) ))
	v0n = np.sqrt( np.sqrt( ( -1.e-5*gnabar*m*m*m*h*(vx-ena)- 1.e-5*gl*(vx- el)+(-inoise*1e-3-gsyn*(vx- es)*1e-3)/s ) /
		( 1.e-5 * gkbar * (vx- ek) ) ))
	return vx,u0,v0,v0n
	
def numsptk(postidx,idxrange):
	prespikes = np.array([])
	trange=t[idxrange]
	sptk = np.zeros(trange.size)
	for nidx in OUTList[postidx]:
		sptime = np.array(neurons[nidx].spks)
		sptime = sptime[ np.where( (sptime > trange[0]) * (sptime < trange[-1]) ) ]
		prespikes = np.append(prespikes,sptime)
	
	prespikes = np.sort(prespikes)
	#print prespikes
	accumulator = 0
	for tm in trange:
		mp = np.where(prespikes < tm)[0]
		sptk[np.where( trange == tm )] = mp.size
	return sptk
	
def getprespikes(postidx,tl,tr):
	postspk = []
	for nidx in OUTList[postidx]:
		for nspk in neurons[nidx].spks[ np.where( (neurons[nidx].spks >= tl)*(neurons[nidx].spks < tr) ) ]:
			postspk.append([nspk,nidx] )
		
	return np.array( postspk )

def zoolyupdate(imax):
	zoolyclickevent.spikesymbol = "."
	zoolyclickevent.imax = imax
	onclick1.lines[imax].set_linewidth(4)
	onclick1.lines[imax].set_ls("--")
	zooly.canvas.draw()
	
	zoolyclickevent.v = np.array(neurons[imax].volt)
	zoolyclickevent.u = np.array(neurons[imax].svar)
	zoolyclickevent.g = np.array(neurons[imax].isyng)
	zoolyclickevent.i = np.array(neurons[imax].inoise)
	zoolyclickevent.rst = getprespikes(imax,onclick1.tl, onclick1.tr)
	moddyupdate(idx)
	
def zoolyclickevent(event):
	if not hasattr(onclick1,"lines"): return
	et = event.xdata
	ev = event.ydata
	idx = np.where( np.abs(t-et)<h.dt)[0][0]
	#DB>>
	#print idx, et,ev
	#<<DB
	vmax = abs(neurons[0].volt.x[idx] - ev)
	zoolyclickevent.imax = 0
	for ind,n in map(None,xrange(methods["ncell"]),neurons):
		onclick1.lines[ind].set_linewidth(1)
		onclick1.lines[ind].set_ls("-")
		if vmax > abs(n.volt.x[idx] - ev) :
			vmax = abs(n.volt.x[idx] - ev)
			zoolyclickevent.imax = ind
		#print vmax,n.volt.x[idx],ev

def moddyclickevent(event):
	et = event.xdata
	idx = np.where( np.abs(t-et)<h.dt)[0][0]
	moddyupdate(idx)


def moddyupdate(idx):
	if not hasattr(moddyupdate,"tail"):
		moddyupdate.tail = False
	if moddyupdate.tail:
		ridx = np.where( onclick1.idx == idx)[0][0]+1
		moddyupdate.tailsize = 300
		lidx = ridx-moddyupdate.tailsize
		if lidx < 0 :
			lidx = 0
	moddyupdate.idx = idx
	vmin,vmax =-85,+40
	vx,u0,v0,v0n = getnulls(vmin,vmax,zoolyclickevent.g[idx], zoolyclickevent.i[idx],neurons[zoolyclickevent.imax].innp.mean)
	moddyupdate.rst  = getprespikes(zoolyclickevent.imax,onclick1.tl, onclick1.tr)
	###!!!!zoolyclickevent.lines

	if not hasattr(moddyupdate,"lines"):
		
		dsynmax = np.max(zoolyclickevent.g[onclick1.idx]) if not checkinmethods("tV-synmax") else methods["tV-synmax"]
		#DB>>
		print  "\n\n----\nDB! dsynmax",dsynmax,"\n----\n\n"
		#<<DB
		moddyupdate.lines = [
			faxi.plot(zoolyclickevent.rst[:,0],zoolyclickevent.rst[:,1],"k"+zoolyclickevent.spikesymbol,ms=9,lw=5)[0],
			vaxi.plot(tprin[onclick1.idx],zoolyclickevent.v[onclick1.idx],"k-")[0],
			uaxi.plot(tprin[onclick1.idx],zoolyclickevent.u[onclick1.idx],"k-")[0],
			gaxi.plot(tprin[onclick1.idx],zoolyclickevent.g[onclick1.idx],"k-")[0],
			iaxi.plot(tprin[onclick1.idx],zoolyclickevent.i[onclick1.idx],"k-")[0],
			naxi.scatter(zoolyclickevent.v[onclick1.idx],zoolyclickevent.u[onclick1.idx],\
				c=zoolyclickevent.g[onclick1.idx]/dsynmax,cmap=cmap,vmin=0., vmax=1.,linewidths=0)\
				if checkinmethods("color-phase") else\
			naxi.plot(zoolyclickevent.v[onclick1.idx],zoolyclickevent.u[onclick1.idx],"k-")[0],
			naxi.plot(vx,u0,"r:",label="n\'=0")[0],
			naxi.plot(vx,v0,"b-.",label="v\'=0",lw=2)[0],
			None if checkinmethods("non-isnt-vnul") else naxi.plot(vx,v0n,"b.",mfc="b",mec="b",ms=9)[0],
			naxi.plot([zoolyclickevent.v[idx]],[zoolyclickevent.u[idx]],"r.",mfc="r",mec="r",ms=24)[0]
		]
		try:
			with open("nulls/threshould-JR.pkl",'rb') as fd:
				thx = pickle.load(fd)
				moddyupdate.lines.append(naxi.plot(thx[:,0],thx[:,1],"g--",label="dv/dn=1"),)
		except:
			pass
		naxi.legend(loc=0)
	else:
		moddyupdate.lines[0].set_xdata(zoolyclickevent.rst[:,0])
		moddyupdate.lines[0].set_ydata(zoolyclickevent.rst[:,1])
		moddyupdate.lines[1].set_xdata(tprin[onclick1.idx])
		moddyupdate.lines[1].set_ydata(zoolyclickevent.v[onclick1.idx])
		moddyupdate.lines[2].set_xdata(tprin[onclick1.idx])
		moddyupdate.lines[2].set_ydata(zoolyclickevent.u[onclick1.idx])
		moddyupdate.lines[3].set_xdata(tprin[onclick1.idx])
		moddyupdate.lines[3].set_ydata(zoolyclickevent.g[onclick1.idx])
		moddyupdate.lines[4].set_xdata(tprin[onclick1.idx])
		moddyupdate.lines[4].set_ydata(zoolyclickevent.i[onclick1.idx])
		##
		if checkinmethods("color-phase"):pass
		elif moddyupdate.tail:
			moddyupdate.lines[5].set_xdata(zoolyclickevent.v[onclick1.idx[lidx:ridx]])
			moddyupdate.lines[5].set_ydata(zoolyclickevent.u[onclick1.idx[lidx:ridx]])
		else:
			moddyupdate.lines[5].set_xdata(zoolyclickevent.v[onclick1.idx])
			moddyupdate.lines[5].set_ydata(zoolyclickevent.u[onclick1.idx])
		moddyupdate.lines[6].set_xdata(vx)
		moddyupdate.lines[6].set_ydata(u0)
		moddyupdate.lines[7].set_xdata(vx)
		moddyupdate.lines[7].set_ydata(v0)
		if not checkinmethods("non-isnt-vnul"):
			moddyupdate.lines[8].set_xdata(vx)
			moddyupdate.lines[8].set_ydata(v0n)
		moddyupdate.lines[9].set_xdata([zoolyclickevent.v[idx]])
		moddyupdate.lines[9].set_ydata([zoolyclickevent.u[idx]])
	faxi.set_ylim(0,methods["ncell"])
	vaxi.set_ylim(-85.,40.)
	uaxi.set_ylim(0.,1.)
	if not checkinmethods("tV-synmax"):
		gaxi.set_ylim(0.,zoolyclickevent.g[onclick1.idx].max())
	else:
		gaxi.set_ylim(0.,methods["tV-synmax"])
	iaxi.set_ylim(zoolyclickevent.i[onclick1.idx].min(),zoolyclickevent.i[onclick1.idx].max())
	faxi.set_xlim(onclick1.tl, onclick1.tr)
	naxi.set_xlim(vmin,vmax)
	naxi.set_ylim(0.,1.)
	if not hasattr(moddyupdate,"markers"):
		moddyupdate.markers= [
			faxi.plot([t[idx],t[idx]],[0,methods['ncell']],"r--")[0],
			vaxi.plot([t[idx],t[idx]],[vmin,vmax],"r--")[0],
			uaxi.plot([t[idx],t[idx]],[0,1],"r--")[0],
			gaxi.plot([t[idx],t[idx]],[0,zoolyclickevent.g[onclick1.idx].max()],"r--")[0],
			iaxi.plot([t[idx],t[idx]],iaxi.get_ylim(),"r--")[0]
		]

	else:
		for line in moddyupdate.markers:
			line.set_xdata([t[idx],t[idx]])
	moddy.canvas.draw()

	
def zoolykeyevent(event):
	if not hasattr(zoolyclickevent,"lines"): return
	if event.key == "K":
		v,u,g,i = (
			np.array(neurons[zoolyclickevent.imax].volt),
			np.array(neurons[zoolyclickevent.imax].svar),
			np.array(neurons[zoolyclickevent.imax].isyng),
			np.array(neurons[zoolyclickevent.imax].inoise)
			)
		sptk = numsptk(zoolyclickevent.imax,onclick1.idx)
		rst = getprespikes(zoolyclickevent.imax,onclick1.tl, onclick1.tr)
		moddyupdate.lines.append(faxi.plot(zoolyclickevent.rst[:,0],zoolyclickevent.rst[:,1],zoolyclickevent.spikesymbol,ms=7,lw=5)[0])
		moddyupdate.lines.append(vaxi.plot(tprin[onclick1.idx],v[onclick1.idx])[0])
		moddyupdate.lines.append(uaxi.plot(tprin[onclick1.idx],u[onclick1.idx])[0])
		moddyupdate.lines.append(gaxi.plot(tprin[onclick1.idx],g[onclick1.idx])[0])
		moddyupdate.lines.append(naxi.plot(v[onclick1.idx],u[onclick1.idx])[0])
		moddyupdate.lines.append(iaxi.plot(tprin[onclick1.idx],i[onclick1.idx])[0])
		#zoolyclickevent.lines.append(saxi.plot(tprin[onclick1.idx],sptk)[0])
	elif event.key == "X":
		for lin in moddyupdate.lines:
			lin.remove()
		del zoolyclickevent.lines
	moddy.canvas.draw()	

def moddykeyevent(event):
	if event.key == "K" or event.key == "X":
		zoolykeyevent(event)
	elif event.key == "left":
		moddyupdate.idx -= 1
		if moddyupdate.idx < onclick1.idx[0] :
			moddyupdate.idx = onclick1.idx[0]
		moddyupdate(moddyupdate.idx)
	elif event.key == "right":
		moddyupdate.idx += 1
		if moddyupdate.idx > onclick1.idx[-1] :
			moddyupdate.idx = onclick1.idx[-1]
		moddyupdate(moddyupdate.idx)
	elif event.key == "pageup":
		moddyupdate.idx -= 10
		if moddyupdate.idx < onclick1.idx[0] :
			moddyupdate.idx = onclick1.idx[0]
		moddyupdate(moddyupdate.idx)
	elif event.key == "pagedown":
		moddyupdate.idx += 10
		if moddyupdate.idx > onclick1.idx[-1] :
			moddyupdate.idx = onclick1.idx[-1]
		moddyupdate(moddyupdate.idx)
	elif event.key == "home":
		moddyupdate.idx = onclick1.idx[0]
		moddyupdate(moddyupdate.idx)
	elif event.key == "end":
		moddyupdate.idx = onclick1.idx[-1]
		moddyupdate(moddyupdate.idx)
	elif event.key == "T":
		moddyupdate.tail = not moddyupdate.tail
		moddyupdate(moddyupdate.idx)
	elif event.key == "M":
		ridx = np.where( onclick1.idx == moddyupdate.idx)[0][0]+1
		nmax = len(onclick1.idx[ridx::5])
		moddy.set_size_inches(18.5, 10.5, forward=True)
		timestamp = time.strftime("%Y%m%d%H%M%S")
		print "=================================="
		print "===        Making MOVIE        ==="
		print "  > Time Stamp                 : ",timestamp
		print "  > Fraim step (mc)            : ",5. * methods['timestep']
		print "  > Tail length (ms)           : ",float(moddyupdate.tailsize) * methods['timestep']
		for ndx,idx in enumerate(onclick1.idx[ridx::5]):
			moddyupdate(idx)
			#moddy.savefig("/home/rth/media/rth-storage-old/rth/tmp/%s-%04d.jpg"%(timestamp,ndx))
			moddy.savefig("/home/rth/tmp/movies/%s-%04d.jpg"%(timestamp,ndx))
			sys.stderr.write("  > Frame:%04d of %04d         : Saved\r"%(ndx,nmax))
		print "\n==================================\n"
	elif event.key == "S":
		if hasattr(moddykeyevent,"spx"):
			spx.remove()
			del spx
		else:
			with open("separatrix/separatrix.pkl",'rb') as fd:
				spx = pickle.load(fd)
			for fx in np.linspace(0.0,0.27,6):
				naxi.fill_between(spx[:,0],spx[:,1]+fx, spx[:,1]-fx, facecolor='grey', alpha=0.3-fx*0.3/0.27)
			
			spx, = naxi.plot(spx[:,0],spx[:,1],"k--")
		moddy.canvas.draw()	
	#elif event.key == "J":
		#if hasattr(moddykeyevent,"thx"):
			#thx.remove()
			#del thx
		#else:
			##with open("nulls/threshould.pkl",'rb') as fd:
			#with open("nulls/threshould-JR.pkl",'rb') as fd:
				#thx = pickle.load(fd)
			#print thx
			#spx, = naxi.plot(thx[:,0],thx[:,1],"g--")
		#moddy.canvas.draw()	
	elif event.key == "D":
		if hasattr(moddykeyevent,"d10p"):
			for x in moddykeyevent.d10p: x.remove()
			del moddykeyevent.d10p
		else:
			vdata = np.array(neurons[zoolyclickevent.imax].volt)
			udata = np.array(neurons[zoolyclickevent.imax].svar)
			gdata = np.array(neurons[zoolyclickevent.imax].isyng)
			moddykeyevent.d10p = []
			maxg = methods['maxg'] if checkinmethods('maxg') else np.max(gdata)*0.1 
			print "MAXG:",maxg
			d10idx = np.where(gdata<maxg)[0]
			d10idx = [ idx0 for idx0, idx1 in zip(d10idx[:-1],d10idx[1:]) if idx1 != idx0+1 and idx0>=onclick1.idx[0] and idx0<=onclick1.idx[-1]]+\
			         [ idx1 for idx0, idx1 in zip(d10idx[:-1],d10idx[1:]) if idx0 != idx1-1 and idx1>=onclick1.idx[0] and idx1<=onclick1.idx[-1]]
			d10idx.sort()
			moddykeyevent.d10p.append( gaxi.plot(tprin[d10idx], gdata[d10idx], "kX",ms=12)[0] )
			moddykeyevent.d10p.append( naxi.plot(vdata[d10idx], udata[d10idx], "kX",ms=12)[0] )
			
			d10idx = (np.diff(np.sign(np.diff(gdata))) < 0).nonzero()[0] + 1
			d10idx = [ idx for idx in d10idx if idx>=onclick1.idx[0] and idx<=onclick1.idx[-1]]
			d10idx.sort()
			moddykeyevent.d10p.append( gaxi.plot(tprin[d10idx], gdata[d10idx], "rX",ms=12)[0] )
			moddykeyevent.d10p.append( naxi.plot(vdata[d10idx], udata[d10idx], "rX",ms=12)[0] )
		moddy.canvas.draw()
	elif  event.key == "G":
		print np.max(np.array(neurons[zoolyclickevent.imax].isyng) )
	else:
		print event.key
		

def neuronsoverview(event):
	global vindex
	if event.key == "up":
		vindex += 1
		if vindex >= methods["ncell"] : vindex = methods["ncell"] -1
	elif event.key == "down":
		vindex -= 1
		if vindex < 0 : vindex = 0
	elif event.key == "home": vindex = 0
	elif event.key == "end" : vindex = methods["ncell"] -1
	if event.key == "pageup":
		vindex += 10
		if vindex >= methods["ncell"] : vindex = methods["ncell"] -1
	elif event.key == "pagedown":
		vindex -= 10
		if vindex < 0 : vindex = 0
	vtrace.set_ydata( np.array(neurons[vindex].volt)[tproc:tproc+tprin.size])
	if methods['tracetail'] == 'conductance':
		xvcrv.set_ydata( np.array(neurons[vindex].isyng)[tproc:tproc+tprin.size] )
	elif methods['tracetail'] == 'current':
		xvcrv.set_ydata( np.array(neurons[vindex].isyni)[tproc:+tprin.size] )
	mainfig.canvas.draw()
	if checkinmethods('traceView'):
		zoolyupdate(vindex)
		moddyupdate(moddyupdate.idx)

def positiveGauss(mean,stdev):
	result = -1
	while result < 0:
		result = mean + np.random.randn()*stdev
	return result

def checkinmethods(item, dirtree = methods):
	def getsubitems(item):
		items = item.split("/")
		if items[ 0] == "" and len(items) !=1 : items = items[1:]
		if items[-1] == "" and len(items) !=1 : items = items[:-1]
		return items[0],"/".join(items[1:])
	item,subitems = getsubitems(item)
	if subitems != "":
		if not item in dirtree : return False
		if not type(dirtree[item]) is dict: return False
		return checkinmethods(subitems,dirtree[item])
	else:
		if not item in dirtree : return False
		if not ( (type(dirtree[item]) is bool or type(dirtree[item]) is int) ):
			if dirtree[item] is None: return False
			else: return True
		return bool(dirtree[item])

def ggap_var(t,t0,t1,r0,r1):
	if t < t0:
		for gj in gapjuctions: gj[0].r, gj[1].r = r0, r0
	elif t > t1:
		for gj in gapjuctions: gj[0].r, gj[1].r = r1, r1
	else :
		r = (r1-r0)*(t-t0)/(t1-t0)+r0
		for gj in gapjuctions: gj[0].r, gj[1].r = r, r
	#DB>>
#	print "ggap_var was called with parameters", t,t0,t1,r0,r1
#	exit(0)
	#<<DB
	
			
	
#===============================================#
#               MAIN PROGRAMM                   #
#===============================================#
if __name__ == "__main__":
	if len(sys.argv) > 1:
		def setmethod(arg):
			global methods
			if not "=" in arg: return
			name,value = arg.split("=")
			if not "/" in name: return
			if name[0] != '/' : return
			names = name.split("/")
			if len(names) > 2:
				name = "methods"
				for nm in names[1:-1]:
					exec "inmethods=\'"+nm+"\' in "+name
					if inmethods :
						exec "inmethods=type("+name+"[\'"+nm+"\']) is dict"
						if inmethods :
							name += "[\'"+nm+"\']"
							continue
						else:
							exec "inmethods=type("+name+"[\'"+nm+"\']) is bool or type("+name+"[\'"+nm+"\']) is None"
							if inmethods :
								name += "[\'"+nm+"\']"
								exec name+"={}"
							else:
								sys.stderr.write("method item %s of %s isn't dict\n"%(name,nm))
					else:
						name += "[\'"+nm+"\']"
						exec name+"={}"
					
			cmd = "methods" + reduce(lambda x,y: x+"[\'"+y+"\']", names[1:],"") + "=" + value
			try:
				exec cmd
			except: 
				#cmd = "methods" + reduce(lambda x,y: x+"[\'"+y+"\']", names[1:],"") + "=" + "\'\\\'"+value+"\\\'\'"
				cmd = "methods" + reduce(lambda x,y: x+"[\'"+y+"\']", names[1:],"") + "=" + "\'"+value+"\'"
				exec cmd
		def readfromsimdb(simdb,ln):
			rec = None
			with open(simdb) as fd:
				for il,l in enumerate(fd.readlines()):
					if il == ln: rec = l
			if rec == None:
				sys.stderr.write("Cannot find line %d in the file %s\n"%(ln,simdb))
				exit(1)
			for itm in rec.split(":"):
				#DB>>
				print itm
				#<<DB
				setmethod(itm)
		simdbrec=None
		for arg in sys.argv:
			if arg[:len('--readsimdb=')] == '--readsimdb=':
				simdbrec=arg[len('--readsimdb='):]
			if arg == '-h' or arg == '-help' or arg == '--h' or arg == '--help':
				print __doc__
				print 
				print "USAGE: nrngui -nogui -python network.py [parameters]"
				print "\nPARAMETERS:"
				print '   /ncell = <int>                           - a number of neurons in the netwrok'
				print
				print '   /ncon =<int|(ftom,to)|...>               - a number of input connections per neuron'
				print '           it may be a constant or uniform distribution(from, to) or normalized uniform distribution (from, to, ncon-norm)'
				print '           OR uniform distribution  (\\\'u\\\', from, {to},    {{ncon-norm}})'
				print '           OR normal distribution   (\\\'n\\\', mean, {stdev}, {{ncon-norm}})'
				print '           OR binomial distribution (\\\'b\\\', prob,           {ncon-norm} )'
				print
				print '   /neuron/Vinit=<float|(mean,stdev)|[list for each neuron]|filename where to read>'
				print '                                            - initial conditions for voltage'
				print '   /neuron/Iapp=<float|(mean,stdev)|None>   - applay current for each neuron a constant or mean,stdev or None to turn off'
				print '   /neuron/Istdev=<float|(mean,stdev)|None> - standard deviation for noise current'
				print '           it may be a  constant or mean,stdev or None to turn off'
				print
				print '   /synapse/weight=<float|(mean,std)>       - a synaptic conductance: a constant or mean,stdev'
				print '   /synapse/delay=<float|(mean,std)|(mean,std,tr)>'
				print '                                            - set axonal delay with truncation if needed'
				print '   /synapse/gsynscale=<float|(mean,stdev)>  - set distribution of total synaptic conductance as a mutiplier to weight'
				print '   /synapse/tau1=<float>                    - tau 1 of synaptic constant'
				print '   /synapse/tau2=<float>                    - tau 2 of synaptic constant'
				print '   /synapse/Esyn=<float|(mean,stdev)|[list for all neurons|filename to read>'
				print '                                            - synaptic reversal potentials'
				print
				print '   /gsyn-dist=<NORM|LOGN>                   - set distribution for synaptic conductqnce'
				print '   /delay-dist=<NORM|LOGN|LOGN_SHIFT|DIST>  - set distribution of delay.'
				print '   /gtot-dist=<NORM|LOGN>                   - set distribution for total synaptic conductance'
				print
				print '   /R2=<True|False>                         - calualte R2'
				print '   /T&S=<True|False>                        - turn on/off Tiesinga and Sejnowski synchronization index'
				print '   /lastspktrg=<True|False>                 - turn on/off detector of sustanable firing rate'
				print '   /maxFreq=<float>                         - max frequency fft'
				print '   /netFFT=<True|False>                     - turn on/off network FFT'
				print '   nrnFFT=<True|False>                      - turn on/off neuron FFT'
				print '   /netISI=<maxISI>                         - perform network ISI analysis for ISI<maxISI'
				print '   /nrnISI=<maxISI>                         - perform neurons ISI analysis for ISI<maxISI'
				print '   /gui=<True|False>                        - turn on/off gui'
				print '   /git=<True|False>                        - trun on/off git recording'
				print
				print '   /Gtot-dist=<True|False>                  - Get a statistics of total conductance distribution'
				print '   /Gtot-stat=<True|False>                  - the same but with any distribution'
				print
				print '   /tstop=<float>                           - model time to stop simulations'
				print '   /timestep=<float>                        - set simulation time step'
				print '   /sortbysk=<False|ST|GT|N|I|NC>           - sort raster diagram'
				print '   /preview=<True|False>                    - turn on/off preview'
				print '   /tv=(0.,tstop)                           - preview only this interval'
				print
				print ' --- additional noise ---'
				print '   /nstart=False                            - turns it off'
				print '   /nstart/Istdev=<float>                   - standard deviation of noise '
				print '   /nstart/delay=<float>                    - noise begins at this time'
				print '   /cliprst=<float>                         - remove first n-millisecond from analysis'
				exit(0)
			
		if not simdbrec is None:
			simdbrec = simdbrec.split(":")
			if len(simdbrec) < 2:
				sys.stderr.write("Error format --readsimdb=file:record\n")
			readfromsimdb( simdbrec[0], int(simdbrec[1]) ) 
		for arg in sys.argv: setmethod(arg)
			

	if 'cmd-file' in methods:
		if not type(methods['cmd-file']) is str: methods['cmd-file'] = 'network.start'
	else:
		methods['cmd-file'] = 'network.start'
	with open(methods['cmd-file'],"w") as fd:
		for arg in sys.argv: fd.write("%s "%arg)
	if (methods['synapse']['tau1'] != 2.0 or methods['synapse']['tau2'] != 5.0) and methods['taunorm']:
		from norm_translation import getscale
		nFactor = getscale(2.0,5.0,methods['synapse']['tau1'],methods['synapse']['tau2'])
		if type(methods["synapse"]["weight"]) == tuple or type(methods["synapse"]["weight"]) == list:
			methods["synapse"]["weight"]		= (methods["synapse"]["weight"][0]*nFactor, methods["synapse"]["weight"][1]*nFactor)
		else:
			weight		= (methods["synapse"]["weight"]*nFactor, 0.)
	if checkinmethods('preview'):
		methods['tstop'] = methods['tv'][1]

###DB>
	print "=================================="
	print "==       ::  METHODS  ::        =="
	def dicprn(dic, space):
		for nidx,name in enumerate(sorted([ x for x in dic ])):
			if type(dic[name]) is dict:
				rep = "%s%s\\ %s "%(space,"v-" if nidx==0 else "|-", name)
				print rep
				dicprn(dic[name], space+"  ")
			else:
				rep = "%s%s> %s "%(space,"`-" if nidx==len(dic)-1 else "|-", name)
				if len(rep) < 31:
					for x in xrange(31-len(rep)):rep += " "
				if type(dic[name]) is str:
					print rep," : ","\'%s\'"%dic[name]
				else:
					print rep," : ",dic[name]
	dicprn(methods,' ')
	print "==================================\n"
###<DB

	
	if methods["gui"]:
		import matplotlib
		import matplotlib.pyplot as plt
		matplotlib.rcParams["savefig.directory"] = ""
		#cmap = matplotlib.cm.get_cmap('jet')
		#cmap = matplotlib.cm.get_cmap('plasma')
		#cmap = matplotlib.cm.get_cmap('autumn')
		#cmap = matplotlib.cm.get_cmap('gist_rainbow')
		cmap = matplotlib.cm.get_cmap('rainbow')
		print "=================================="
		print "===        GUI turned ON       ==="
		print "==================================\n"
	
	h.tstop 	= float(methods['tstop'])
	#h.v_init 	= V
	h.dt		= float(methods["timestep"])

	if checkinmethods('simvar'):
		if not type(methods['simvar']) is dict: methods['simvar'] = False
		elif not "type" in methods['simvar']: methods['simvar'] = False
		elif not type(methods['simvar']["type"]) is str: methods['simvar'] = False
		elif not (methods['simvar']["type"] == 'n' or methods['simvar']["type"] == 'c' or methods['simvar']["type"] == 'g'  ): methods['simvar'] = False
		elif not "var" in methods['simvar']: methods['simvar'] = False
		elif not type(methods['simvar']["var"]) is str: methods['simvar'] = False
		elif not "a0" in methods['simvar']: methods['simvar'] = False
		elif not (type(methods['simvar']["a0"]) is float or type(methods['simvar']["a0"]) is int): methods['simvar'] = False
		elif not "a1" in methods['simvar']: methods['simvar'] = False
		elif not (type(methods['simvar']["a1"]) is float or type(methods['simvar']["a1"]) is int): methods['simvar'] = False
		elif not "t0" in methods['simvar']: methods['simvar']['t0'] = methods['tv'][0]
		elif not (type(methods['simvar']["t0"]) is float or type(methods['simvar']["t0"]) is int): methods['simvar'] = False
		elif not "t1" in methods['simvar']: methods['simvar']['t1'] = methods['tv'][1]
		elif not (type(methods['simvar']["t1"]) is float or type(methods['simvar']["t1"]) is int): methods['simvar'] = False	

	#### Save mamory, record only what is needed
	if checkinmethods('gui'):
		print "=================================="
		print "===          RECORDER          ==="

		methods['neuron']['record'] = {}
		if methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'mean total conductance' or\
		   methods['tracetail'] == 'TG'                or methods['tracetail'] == 'MTG'                    or\
		   methods['tracetail'] == 'conductance'       or\
		   checkinmethods('traceView'):
			   methods['neuron']['record']['conductance'] = True
			   print " > RECORD                       : cunductance"
		
		if methods['tracetail'] == 'total current'               or methods['tracetail'] == 'TI'   or\
		   methods['tracetail'] == 'mean total current'          or methods['tracetail'] == 'MTI'  or\
		   methods['tracetail'] == 'total synaptic current'      or methods['tracetail'] == 'TSI'  or\
		   methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI' or\
		   methods['tracetail'] == 'current'                     or\
		   checkinmethods('spectrogram'):
			   methods['neuron']['record']['current'] = True
			   print " > RECORD                       : current"
		print "==================================\n"
	
	#### Create Neurons and setup noise and Iapp
	print "=================================="
	print "===        Create Neurons      ==="
	neurons = [ neuron() for x in xrange(methods["ncell"]) ]
	
	if type(methods["neuron"]["Vinit"]) is float or type(methods["neuron"]["Vinit"]) is int:
		xV = methods["neuron"]["Vinit"]*np.ones(methods["ncell"])
	elif type(methods["neuron"]["Vinit"]) is tuple:
		xV = methods["neuron"]["Vinit"][0]+np.random.randn(methods["ncell"])*methods["neuron"]["Vinit"][1]
	elif type(methods["neuron"]["Vinit"]) is list:
		if len(methods["neuron"]["Vinit"]) == methods['ncell'] :
			xV = np.array(methods["neuron"]["Vinit"])
		else: 
			xV = [ None for i in xrange(methods["ncell"]) ]
	elif type(methods["neuron"]["Vinit"]) is str:
		xV = np.genfromtxt(methods["neuron"]["Vinit"])
		print "  > Read Vinit from file         :",methods["neuron"]["Vinit"]
	if methods["neuron"]["Vinit"] == None:
		xV = [ None for i in xrange(methods["ncell"]) ]
	if type(methods['synapse']['Esyn']) is float or type(methods['synapse']['Esyn']) in int:
		xEsyn = float( methods['synapse']['Esyn'] )*np.ones(methods["ncell"])
	elif type(methods['synapse']['Esyn']) is tuple:
		if len(methods['synapse']['Esyn']) >=2 :
			xEsyn = methods['synapse']['Esyn'][0] + np.random.randn(methods["ncell"]) * methods['synapse']['Esyn'][1]
		else:
			xEsyn = float(methods['synapse']['Esyn'][0])*np.ones(methods["ncell"])
	elif type(methods['synapse']['Esyn']) is list:
		xEsyn = methods['synapse']['Esyn']
	elif type(methods['synapse']['Esyn']) is str:
		xEsyn = np.genfromtxt(methods['synapse']['Esyn'])

	if type(methods["neuron"]["Iapp"]) is float or type(methods["neuron"]["Iapp"]) is int:
		xIapp = np.ones(methods["ncell"])*float(methods["neuron"]["Iapp"]) * (-1)
	elif type(methods["neuron"]["Iapp"]) is tuple:
		xIapp = (methods["neuron"]["Iapp"][0]+np.random.randn(methods["ncell"])*methods["neuron"]["Iapp"][1]) * (-1)
	elif type(methods["neuron"]["Iapp"]) is list:
		xIapp = np.array(methods["neuron"]["Iapp"]) * (-1)
	elif type(methods["neuron"]["Iapp"]) is str:
		xIapp = np.genfromtxt(methods["neuron"]["Iapp"]) * (-1)
	elif methods["neuron"]["Iapp"] == None:
		xIapp = [ None for i in xrange(methods["ncell"]) ]
			
	if type(methods["neuron"]["Istdev"]) is float or type(methods["neuron"]["Istdev"]) is int:
		xIstdev = float(methods["neuron"]["Istdev"]) * np.ones(methods["ncell"])
	elif type(methods["neuron"]["Istdev"]) is tuple:
		xIstdev = methods["neuron"]["Istdev"][0]+np.random.randn(methods["ncell"])*methods["neuron"]["Istdev"][1]
	elif type(methods["neuron"]["Istdev"]) is list:
		xIstdev = np.array(methods["neuron"]["Istdev"])
	elif type(methods["neuron"]["Istdev"]) is str:
		xIstdev = np.genfromtxt(methods["neuron"]["Istdev"])
	elif methods["neuron"]["Istdev"] == None:
		xIstdev = [ None for i in xrange(methods["ncell"]) ]
			
	for n,i in zip(neurons,xrange(methods["ncell"])):
		if not methods['synapse']['synscaler'] is None:
			if type(methods['synapse']['synscaler']) is float or type(methods['synapse']['synscaler']) is int:
				n.tsynscale = float(methods['synapse']['synscaler'])
				xTau1,xTau2 = methods['synapse']['tau1'] * n.tsynscale, methods['synapse']['tau2'] * n.tsynscale
			elif type(methods['synapse']['synscaler']) is list or type(methods['synapse']['synscaler']) is tuple:
				if len(methods['synapse']['synscaler']) >= 2:
					n.tsynscale = -1.0
					while( n.tsynscale < 0.0 ):
						n.tsynscale = methods['synapse']['synscaler'][0] + np.random.randn()*methods['synapse']['synscaler'][1]
					xTau1,xTau2 = methods['synapse']['tau1'] * n.tsynscale, methods['synapse']['tau2'] * n.tsynscale
				else :
					n.tsynscale = float(methods['synapse']['synscaler'][0])
					xTau1,xTau2 = methods['synapse']['tau1'] * n.tsynscale, methods['synapse']['tau2'] * n.tsynscale
			else:
				xTau1,xTau2 = methods['synapse']['tau1'],methods['synapse']['tau2']
		else:
			xTau1,xTau2 = methods['synapse']['tau1'],methods['synapse']['tau2']

		n.setparams(
			V=xV[i],
			SynT1=xTau1, SynT2=xTau2, SynE=xEsyn[i], 
			Iapp = xIapp[i], Insd=xIstdev[i]
			)
	print "==================================\n"
							
		

	if checkinmethods('nstart'):
		if type(methods['nstart']) is list or type(methods['nstart']) is tuple:
			methods['nstart'] = {
				'delay'    : methods['nstart'][0],
				'Istdev'   : methods['nstart'][1],
				'duration' : methods['nstart'][2],
			}
		if not checkinmethods('nstart/Iapp'     ): methods['nstart']['Iapp'    ] = 0.
		if not checkinmethods('nstart/period'   ): methods['nstart']['period'  ] = 0.1
		if not checkinmethods('nstart/delay'    ): methods['nstart']['delay'   ] = 0.
		if not checkinmethods('nstart/duration' ): methods['nstart']['duration'] = 1e9
			
		if not checkinmethods('nstart/Istdev'):
			raise RuntimeError("/nstart/Istdev isn't set up")
		for n in neurons:
			n.addnoise(\
				Iapp  = methods['nstart']['Iapp'],\
				Insd  = methods['nstart']['Istdev'],\
				delay = methods['nstart']['delay'],\
				dur   = methods['nstart']['duration'],\
				per   = methods['nstart']['period'] )


	#DB>>
	#for n in neurons:
		#print n.soma(0.5).v
	#exit(0)
	#<<DB

	t = h.Vector()
	t.record(h._ref_t)


	#### Create Connection List:
	if checkinmethods("ncon"):
		def CreateConnectionList():
			def CreateFixNumberOrRange(n0,n1=None):
				OUTList = [ [] for x in xrange(methods["ncell"])]
				for toid in xrange(methods["ncell"]):
					from_excaption = [ 0 for x in xrange(methods["ncell"]) ]
					from_excaption[toid] = 1
					upcnt = 0
					total = 0
					if not n1 is None:
						neurons[toid].concnt = int(np.random.random()*(n1-n0) + n0)
					else:
						neurons[toid].concnt = n0
					while  upcnt < 10000*methods["ncell"]:
						upcnt += 1
						fromid = rnd.randint(0, methods["ncell"]-1)
						if from_excaption[fromid] == 1 : continue
						upcnt  = 0
						total += 1
						from_excaption[fromid] = 1
						OUTList[toid].append(fromid)
						if total >= neurons[toid].concnt :break
					else:
						sys.stderr.write("Couldn't obey connections conditions\nNeuron:%d TOTLA:%d CURRENT:%d\n"%(toid,n0,total))
						for x in OUTList:
							sys.stderr.write("ID:%d #%d\n"%(x[0],x[1]))
						sys.exit(1)
				return OUTList
			def CreateNormalDistribution(mean,stdev=0.):
				OUTList = [ [] for x in xrange(methods["ncell"])]
				for toid in xrange(methods["ncell"]):
					from_excaption = [ 0 for x in xrange(methods["ncell"]) ]
					from_excaption[toid] = 1
					upcnt = 0
					total = 0
					neurons[toid].concnt = int( positiveGauss(mean,stdev) )
					while  upcnt < 10000*methods["ncell"]:
						upcnt += 1
						fromid = rnd.randint(0, methods["ncell"]-1)
						if from_excaption[fromid] == 1 : continue
						upcnt  = 0
						total += 1
						from_excaption[fromid] = 1
						OUTList[toid].append(fromid)
						if total >= neurons[toid].concnt :break
					else:
						sys.stderr.write("Couldn't obey connections conditions\nNeuron:%d TOTLA:%d CURRENT:%d\n"%(toid,n0,total))
						for x in OUTList:
							sys.stderr.write("ID:%d #%d\n"%(x[0],x[1]))
						sys.exit(1)
				return OUTList
			def CreateBinomialDistribution(prob):
				OUTList = [ [] for x in xrange(methods["ncell"])]
				for toid in xrange(methods["ncell"]):
					for fromid in xrange(methods["ncell"]):
						if fromid == toid: continue
						if np.random.random() > prob : continue
						OUTList[toid].append(fromid)
						neurons[toid].concnt += 1
					#DB>>
					#print OUTList[toid]
					#<<DB
				return OUTList
#>>			
			#DB>>
			#print type(methods["ncon"]),methods["ncon"]
			#<<DB
			if type(methods["ncon"]) is int:
				#DB>>
				#print "ncon - int"
				#<<DB
				return CreateFixNumberOrRange(methods["ncon"])
			elif type(methods["ncon"]) is tuple or type(methods["ncon"]) is list:
				#DB>>
				#print "Ncon tuple or list"
				#<<DB
				if type(methods["ncon"][0]) is int:
					if len(methods["ncon"]) > 2:
						methods["normalize-weight-by-ncon"] = methods["ncon"][2]
						return CreateFixNumberOrRange(methods["ncon"][0],methods["ncon"][1])
					elif len(methods["ncon"]) > 1:
						return CreateFixNumberOrRange(methods["ncon"][0],methods["ncon"][1])
					else:
						return CreateFixNumberOrRange(methods["ncon"][0])
				elif type(methods["ncon"][0]) is str:
					if methods["ncon"][0] == "u":
						#print "  > Uniform Distribution "
						if len(methods["ncon"]) > 3:
							methods["normalize-weight-by-ncon"] = methods["ncon"][3]
							return CreateFixNumberOrRange(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 2:
							return CreateFixNumberOrRange(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 1:
							return CreateFixNumberOrRange(methods["ncon"][1])
						else:
							print "ERROR in ncon parameter:\nUSAGE of uniform distribution: /ncom=('u',n-from, {n-to}, {{norm-by}})"
							sys.exit(1)
				
					if methods["ncon"][0] == "n":
						if len(methods["ncon"]) > 3:
							methods["normalize-weight-by-ncon"] = methods["ncon"][3]
							return CreateNormalDistribution(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 2:
							return CreateNormalDistribution(methods["ncon"][1],methods["ncon"][2])
						elif len(methods["ncon"]) > 1:
							return CreateFixNumberOrRange(methods["ncon"][1])
						else:
							print "ERROR in ncon parameter:\nUSAGE for noormal distribution: /ncom=('n',mean, {stdev}, {{norm-by}})"
							sys.exit(1)

					if methods["ncon"][0] == "b":
						if len(methods["ncon"]) > 2:
							methods["normalize-weight-by-ncon"] = methods["ncon"][1]
							return CreateBinomialDistribution(methods["ncon"][1])
						elif len(methods["ncon"]) > 1:
							return CreateBinomialDistribution(methods["ncon"][1])
						else:
							print "ERROR in ncon parameter:\nUSAGE for binomial distribution: /ncom=('b',prob,{norm-by})"
							sys.exit(1)

		print "=================================="
		print "===        Map Connections     ==="
		if checkinmethods('connectom'):
			print "  > Try to Read Connectom file :", methods['connectom']
			if os.access(methods['connectom'],os.R_OK):
				print "  > File is accessible         : "
				with open(methods['connectom'],"r") as fd:
					xncell = pickle.load(fd)
					xnconn = pickle.load(fd)
					if xncell != methods['ncell'] or xnconn != methods['ncon']:
						print "  > File has different numbers : "
						print "  > n cell                     : ", xncell,"|",methods['ncell']
						print "  > n connection               : ", xnconn,"|",methods['ncon']
						OUTList = CreateConnectionList()
					else:
						print "  > Read Connection Map        : ",
						OUTList = pickle.load(fd)
						for n,cpn in zip(neurons,pickle.load(fd)):
							n.concnt = cpn
						if not checkinmethods("normalize-weight-by-ncon"):
							methods["normalize-weight-by-ncon"] = pickle.load(fd)
						print "Successfully"
					
			elif not os.access(methods['connectom'],os.F_OK):
				print "  > File dos not exist         : try to create"
				OUTList = CreateConnectionList()
				with open(methods['connectom'],"w") as fd:
					pickle.dump(methods['ncell'],fd)
					pickle.dump(methods['ncon'],fd)
					pickle.dump(OUTList,fd) 
					pickle.dump([ n.concnt for n in neurons ],fd)
					if checkinmethods("normalize-weight-by-ncon"):
						pickle.dump(methods["normalize-weight-by-ncon"],fd)
					else:
						pickle.dump(False,fd)
			else:
				print
				print "============= ERROR ============="
				print " > Cannot create file \'{}\' ".format(methods['connectom'])
				print
				exit(0)
		else:
			print " > Generate connections         :",
			OUTList = CreateConnectionList()
			print "Successfully"
		print "==================================\n"

	#DB>
		#for i in OUTList:
			#print len(i)
			#for j in i:	print "%03d"%(j),
			#print
		#sys.exit(0)
	#<DB
	if checkinmethods('cycling'):
		print "=================================="
		print "===      Cycles counting       ==="
		mat=np.matrix( np.zeros((methods["ncell"],methods["ncell"])) )
		for i,vec in map(None,xrange(methods["ncell"]),OUTList):
			mat[i,vec]=1
		kx = []
		for cnt in xrange(methods['cycling']):
			kx.append(np.trace(mat)/methods["ncell"])
			mat = mat.dot(mat)
		print " > Cyclopedic numbers           : ",kx
		print "==================================\n"
		methods['cycling-result'] = kx
		del mat
		
		
	

	#### Create Conneactions:
	if checkinmethods("ncon"):
		print "=================================="
		print "===    Make the Connections    ==="
		print "==================================\n"
		connections = []
		if not checkinmethods("gtot-dist"):  methods["gtot-dist"]  = "NORM"
		if not checkinmethods("gsyn-dist"):  methods["gsyn-dist"]  = "NORM"
		if not checkinmethods("delay-dist"): methods["delay-dist"] = "NORM"
		for x in map(None,xrange(methods["ncell"]),OUTList):
			if type(methods['synapse']['gsynscale']) is tuple :
				#DB>>
				#print "DB: TUPLE"
				#<<DB
				if methods['synapse']['gsynscale'][1] <= 0: gx = methods['synapse']['gsynscale'][0]
				elif methods["gtot-dist"] == "NORM":
					### Trancated normal
					gx = positiveGauss(methods['synapse']['gsynscale'][0],methods['synapse']['gsynscale'][1])
				elif methods["gtot-dist"] == "LOGN":
					### Lognormal
					gsymtotm,gsymtots=methods['synapse']['gsynscale']
					gx = np.random.lognormal(mean=np.log(gsymtotm/np.sqrt(1.+gsymtots**2/gsymtotm**2)),sigma=np.sqrt(np.log(1.+gsymtots**2/gsymtotm**2)))
					#DB>>
					#print "DB: gx=",gx
					#<<DB
			else:
				gx = float(methods['synapse']['gsynscale'])
			neurons[x[0]].gsynscale = gx
			for pre in x[1]:
				if type(methods['synapse']['delay']) is tuple :
					if methods['synapse']['delay'][1] <= 0: dx = float(methods['synapse']['delay'][0])
					elif methods["delay-dist"] == "NORM":
						### Trancated normal
						dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
						if len(methods['synapse']['delay']) < 3:
							while(dx < methods['timestep']*2):
								dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
						else:
							while(dx < methods['synapse']['delay'][2]):
								dx = positiveGauss(methods['synapse']['delay'][0],methods['synapse']['delay'][1])
							
					elif methods["delay-dist"] == "LOGN":
						### Lognormal
						dlym,dlys=methods['synapse']['delay'][0]-methods['timestep']*2.,methods['synapse']['delay'][1]
						if len(methods['synapse']['delay']) < 3:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['timestep']*2
						else:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))
							while dx < methods['synapse']['delay'][2]:
								dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))
					elif methods["delay-dist"] == "LOGN-SHIFT":
						### Lognormal
						dlym,dlys=methods['synapse']['delay'][0]-methods['timestep']*2.,methods['synapse']['delay'][1]
						if len(methods['synapse']['delay']) < 3:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['timestep']*2
						else:
							dx = np.random.lognormal(mean=np.log(dlym/np.sqrt(1.+dlys**2/dlym**2)),sigma=np.sqrt(np.log(1.+dlys**2/dlym**2)))+methods['synapse']['delay'][2]
					elif methods["delay-dist"] == "DIST":
						dmin = methods['synapse']['delay'][0]
						dinciment = methods['synapse']['delay'][1] if len(methods['synapse']['delay']) >= 2 else 0.
						dx = dmin + dinciment*float(abs(pre-x[0]))
				else:
					dx = float(methods['synapse']['delay'])
				if type(methods['synapse']['weight']) is tuple :
					if methods['synapse']['weight'][1] <= 0: wx = methods['synapse']['weight'][0]
					elif methods["gsyn-dist"] == "NORM":
						#### Trancated normal
						wx = methods['synapse']['weight'][1]*np.random.randn()+methods['synapse']['weight'][0]
						while wx < 0.0 : wx = methods['synapse']['weight'][1]*np.random.randn()+methods['synapse']['weight'][0]
					elif methods["gsyn-dist"] == "LOGN":
						### Lognormal
						wm,ws=methods['synapse']['weight']
						wx = np.random.lognormal(mean=np.log(wm/np.sqrt(1.+ws**2/wm**2)),sigma=np.sqrt(np.log(1.+ws**2/wm**2)))
						#wx = np.random.lognormal(mean=np.log(wm**2/np.sqrt(wm**2+ws**2)),sigma=np.sqrt(np.log(1.+ws**2/wm**2)))
				else:
					wx = float(methods['synapse']['weight'])
				#if type(methods["ncon"]) is tuple or type(methods["ncon"]) is list:
					#if len(methods["ncon"]) > 2:
						#wx *= float(methods["ncon"][2])/float(neurons[x[0]].concnt)
				if checkinmethods("normalize-weight-by-ncon"):
					#DB>>
					#print "Norm by Factor",float(methods["normalize-weight-by-ncon"]),float(neurons[x[0]].concnt),float(methods["normalize-weight-by-ncon"])/float(neurons[x[0]].concnt)
					#<<DB
					wx *= float(methods["normalize-weight-by-ncon"])/float(neurons[x[0]].concnt)
				if methods['taunorm'] and not methods['synapse']['synscaler'] is None:
					#DB print "norm by factor",1./neurons[x[0]].tsynscale
					wx /= neurons[x[0]].tsynscale
				#####DB>>
				#print "DB:gx=",gx,"dx=",dx,"wx=",wx
				#####<<DB
				connections.append( (h.NetCon(neurons[pre].soma(0.5)._ref_v,neurons[x[0]].isyn,
						0., dx, gx*wx,
						sec=neurons[pre].soma),pre,x[0]) )
				neurons[x[0]].gtotal += gx*wx
		if checkinmethods('Conn-alter'):
			print "================================================"
			print "===        ALTER CONNECTIONS SETTINGS        ==="
			n_alter = methods['Conn-alter']['n']      if checkinmethods('Conn-alter/n')      else 1
			d_alter = methods['Conn-alter']['delay']  if checkinmethods('Conn-alter/delay')  else 0.8
			w_alter = methods['Conn-alter']['weight'] if checkinmethods('Conn-alter/weight') else 0.1e-2
			alter = range(len(connections))
			for i in xrange(n_alter):
				Xalter = alter[np.random.randint(len(alter))]
				connections[Xalter][0].weight[0] = w_alter
				connections[Xalter][0].delay     = d_alter
				print " > %03d -> %03d                                 : were altered"%(connections[Xalter][1],connections[Xalter][2])
				alter.remove(Xalter)
			print "Total number of connections                   :",len(connections)
			print "Number of altered connections                 :",n_alter
			print "Procentage                                    :",n_alter*100/len(connections)
			print "================================================\n"
				
		if checkinmethods('Conn-rec'):
			methods['Conn-rec-results'] = [ (n[1],n[2],n[0].weight[0],n[0].delay) for n in connections ]
		if checkinmethods('Conn-stat'):
			print "================================================"
			print "===           Connections Statistics         ==="
			statn = np.array( [ float(len(o)) for o in OUTList ] )
			meann,stdrn = np.mean(statn),np.std(statn)
			minin,maxin = np.min(statn), np.max(statn)
			statw = np.array( [ n[0].weight[0] for n in connections ] )
			meanw,stdrw = np.mean(statw),np.std(statw)
			miniw,maxiw = np.min(statw), np.max(statw)
			statd = np.array( [ n[0].delay for n in connections ] )
			meand,stdrd = np.mean(statd),np.std(statd)
			minid,maxid = np.min(statd), np.max(statd)
			methods['Conn-stat-results'] = {
				'ncon': {
					'mean':meann, 'stdr':stdrn, 'min':minin, 'max':maxin
				},
				'weight':{
					'mean':meanw, 'stdr':stdrw, 'min':miniw, 'max':maxiw
				},
				'delay':{
					'mean':meand, 'stdr':stdrd, 'min':minid, 'max':maxid
				}
			}
			print " > Number min / max / mean / stdev / CV       :",minin,"/",maxin,"/",meann,"/",stdrn,"/",stdrn/meann
			print " > Weight min / max / mean / stdev / CV       :",miniw,"/",maxiw,"/",meanw,"/",stdrw,"/",stdrw/meanw
			print " > Delay  min / max / mean / stdev / CV       :",minid,"/",maxid,"/",meand,"/",stdrd,"/",stdrd/meand
			print "================================================\n"
		#DB>>
		#plt.figure(0)
		#w=np.array([c[0].weight[0] for c in connections])
		#print np.mean(w), np.std(w)
		#plt.hist(w,bins=50,range=(0,1e-6))
		#plt.show()
		#exit(0)
		#<<DB
			
	
	#### Create gapjunctions:
	if checkinmethods('gapjunction'):
		if   not 'ncon' in methods['gapjunction']            : GJList = OUTList
		elif methods['gapjunction']['ncon'] is None          : GJList = OUTList
		elif not type(methods['gapjunction']['ncon']) is int : GJList = OUTList
		elif not methods['gapjunction']['ncon'] > 0          : GJList = OUTList
		else:
			GJList = [ [] for x in xrange(methods['ncell'])]
			gjncon = methods['gapjunction']['ncon']
			print "=================================="
			print "===       Map Gap-junctions     ==="
			print "==================================\n"
			for toid in xrange(methods['ncell']):
				from_excaption = [ 0 for x in xrange(methods['ncell']) ]
				from_excaption[toid] = 1
				upcnt = 0
				total = 0
				if type(gjncon) is tuple or type(gjncon) is list:
					if len(gjncon) > 1:
						neurons[toid].g_jcnt = int(np.random.random()*(gjncon[1]-gjncon[0]) + gjncon[0])
					else:
						neurons[toid].g_jcnt = gjncon[0]
				else:
					neurons[toid].g_jcnt = gjncon
				while  upcnt < 10000*methods['ncell']:
					upcnt += 1
					fromid = rnd.randint(0, methods['ncell']-1)
					if from_excaption[fromid] == 1 : continue
					upcnt  = 0
					total += 1
					from_excaption[fromid] = 1
					GJList[toid].append(fromid)
					if total >= neurons[toid].g_jcnt :break
				else:
					sys.stderr.write("Couldn't obey connections conditions\nNeuron:%d TOTLA:%d CURRENT:%d\n"%(toid,gjncon,total))
					for x in GJList:
						sys.stderr.write("ID:%d #%d\n"%(x[0],x[1]))
					sys.exit(1)
		
		print "=================================="
		print "===    Make the Gap-Junctions   ==="
		print "==================================\n"
		gapjuctions = []
		for cellid,gjlst in enumerate(GJList):
			for preidx in gjlst:
				gj0,gj1 = h.gap(0.5, sec=neurons[cellid].soma), h.gap(0.5, sec=neurons[preidx].soma)
				h.setpointer(neurons[preidx].soma(.5)._ref_v, 'vgap', gj0)
				h.setpointer(neurons[cellid].soma(.5)._ref_v, 'vgap', gj1)
				gj0.r, gj1.r = methods['gapjunction']['r'],methods['gapjunction']['r']
				gapjuctions.append( (gj0,gj1,neurons[cellid].soma,neurons[preidx].soma) )

	if checkinmethods('simvar'):
		print "=================================="
		print "===      SIMVAR was found!     ==="
		print "==================================\n"
		simvars = []
		if methods['simvar']['type'] == 'n':
			for n in neurons:
				sv = h.variator(0.5, sec=n.soma)
				exec "h.setpointer(n.soma(0.5)."+methods['simvar']["var"]+", \'var\',sv)"
				simvars.append(sv)
			simvarrec = h.Vector()
			exec "simvarrec.record(neurons[0].soma(0.5)."+methods['simvar']["var"]+")"
		#elif methods['simvar']['type'] == 'c':
			#for n in neurons:
				#sv = h.variator()
				#exec "h.setpointer(n."+methods['simvar']["var"]+", \'var\',sv)"
				#simvars.append(sv)
			#simvarrec = h.Vector()
			#exec "simvarrec.record(neurons[0]."+methods['simvar']["var"]+")"
		elif methods['simvar']['type'] == 'g':
			for g0,g1,s0,s1 in gapjuctions:
				sv = h.variator(0.5, sec=s0)
				#DB>>
				#print "h.setpointer(g0."+methods['simvar']["var"]+", \'var\',sv)"
				#<<DB
				exec "h.setpointer(g0."+methods['simvar']["var"]+", \'var\',sv)"
				simvars.append(sv)
				sv = h.variator(0.5, sec=s1)
				exec "h.setpointer(g1."+methods['simvar']["var"]+", \'var\',sv)"
				simvars.append(sv)
			simvarrec = h.Vector()
			exec "simvarrec.record(gapjuctions[0][0]."+methods['simvar']["var"]+")"
		for sv in simvars:
			sv.a0 = methods['simvar']['a0']
			sv.a1 = methods['simvar']['a1']
			sv.t0 = methods['simvar']['t0']
			sv.t1 = methods['simvar']['t1']
	
	if checkinmethods('external'):
		ex_netstim	= h.NetStim(.5, sec = neurons[0].soma)
		if type(methods['external']) is list:
               #              0      1          2         3     4    5    6    7                8
               #/external=\(Start,interval,spike-count,weight,Esyn,Tau1,Tau2,delay,probability of connections\)
			if len(methods['external']) < 8:
			   methods['external'] = {
					'start'       :methods['external'][0],
					'interval'    :methods['external'][1],
					'count'       :methods['external'][2],
					'weight'      :methods['external'][3],
					'E'           :methods['external'][4],
					'tau1'        :methods['external'][5],
					'tau2'        :methods['external'][6]
			   }
			elif len(methods['external']) < 9:
			   methods['external'] = {
					'start'       :methods['external'][0],
					'interval'    :methods['external'][1],
					'count'       :methods['external'][2],
					'weight'      :methods['external'][3],
					'E'           :methods['external'][4],
					'tau1'        :methods['external'][5],
					'tau2'        :methods['external'][6],
					'delay'       :methods['external'][8]
			   }
			elif len(methods['external']) >= 9:
			   methods['external'] = {
					'start'       :methods['external'][0],
					'interval'    :methods['external'][1],
					'count'       :methods['external'][2],
					'weight'      :methods['external'][3],
					'E'           :methods['external'][4],
					'tau1'        :methods['external'][5],
					'tau2'        :methods['external'][6],
					'delay'       :methods['external'][8],
					'p'           :methods['external'][9]
			   }
			if type(methods['external']['delay']) is list or type(methods['external']['delay']) is tuple:
				if len(methods['external']['delay']) < 2:
					methods['external']['delay'] = methods['external']['delay'][0]
				else:
					methods['external']['delay'] = {
						'mean'  : methods['external']['delay'][0],
						'stdev' : methods['external']['delay'][1]
					}
		if not checkinmethods('external/start'   ): methods['external']['start'   ] = methods["tstop"]/3.
		if not checkinmethods('external/interval'): methods['external']['interval'] = methods["tstop"]/6.
		if not checkinmethods('external/count'   ): methods['external']['count'   ] = 1
		if not checkinmethods('external/E'       ): methods['external']['E'       ] = 0
		if not checkinmethods('external/tau1'    ): methods['external']['tau1'    ] = 0.8
		if not checkinmethods('external/tau2'    ): methods['external']['tau2'    ] = 1.2
		if not checkinmethods('external/weight'  ): methods['external']['weight'  ] = 0.
		if not checkinmethods('external/delay'   ): methods['external']['delay'   ] = 1.
		print "================================================"
		print "===              External Input              ==="
		print " > Start                                      :", methods['external']['start']
		print " > Interval                                   :", methods['external']['interval']
		print " > Count                                      :", methods['external']['count']
		if  checkinmethods('external/p'):
			print " > P                                          :", methods['external']['p']
		print " > Reversal potential                         :", methods['external']['E']
		print " > Tau 1                                      :", methods['external']['tau1']
		print " > Tau 2                                      :", methods['external']['tau2']
		print " > Weight                                     :", methods['external']['weight']
		if checkinmethods('external/delay/mean') or checkinmethods('external/delay/stdev'):
			print " > Delay                                    "
			if checkinmethods('external/delay/mean'):
				print "   > mean                                     :", methods['external']['delay']['mean']
			if checkinmethods('external/delay/stdev'):
				print "   > stdev                                    :", methods['external']['delay']['stdev']
		else:
			print " > Delay                                      :", methods['external']['delay']
		print "================================================\n"
		ex_netstim.start	= methods['external']['start'   ]
		ex_netstim.noise	= 0
		ex_netstim.interval	= methods['external']['interval'] 
		ex_netstim.number	= methods['external']['count']
		ex_syn,ex_netcon = [],[]
		for n in neurons:
			if  checkinmethods('external/p'):
				if rnd.random() > methods['external']['p']: continue
			ex_syn_new = h.Exp2Syn(0.5, sec=n.soma)
			ex_syn_new.e	= methods['external']['E'   ]
			ex_syn_new.tau1	= methods['external']['tau1'] if checkinmethods('external/tau1') else 0.8
			ex_syn_new.tau2	= methods['external']['tau2'] if checkinmethods('external/tau2') else 1.2
			ex_syn.append(ex_syn_new)
			if checkinmethods('external/delay/mean') and checkinmethods('external/delay/stdev'):
				exdelay = -1.0
				while exdelay <= 0.0 : exdelay = methods['external']['delay']['mean']+np.random.randn()*methods['external']['delay']['stdev']
			elif checkinmethods('external/delay/mean'):
				exdelay = methods['external']['delay']['mean']
			else :
				exdelay = methods['external']['delay']
			ex_netcon_new	= h.NetCon(ex_netstim, ex_syn_new, 1,exdelay , methods['external']['weight'], sec = n.soma)
			ex_netcon.append(ex_netcon_new)

	#if checkinmethods("wmod"):
		#print "================================================"
		#print "===            Weight Modulator              ==="
		#if not checkinmethods("wmod/scale"          ):methods["wmod"]["scale"        ] = 2.
		#if not checkinmethods("wmod/time-points"    ):methods["wmod"]["time-points"  ] = [0., methods['tstop']]
		#if not checkinmethods("wmod/weight-points"  ):methods["wmod"]["weight-points"] = [ methods["synapse"]["weight"], methods["wmod"]["scale"]*methods['synapse']["weight"] ]
		#wmodT, wmodW = h.Vector(), h.Vector()
		#wmodT.from_python(methods["wmod"]["time-points"  ])
		#wmodW.from_python(methods["wmod"]["wight-points"  ])
		#for c,pre,post in connections:
			#wmodW.play(c._ref_weight[0],wmodT,1)
			

	#if checkinmethods("imod") and checkinmethods("neuron/Iapp"):
		#print "=================================="
		#print "===     Current Modulator      ==="
		#if not checkinmethods("imod/scale"          ):methods["imod"]["scale"         ] = 2.
		#if not checkinmethods("imod/time-points"    ):methods["imod"]["time-points"   ] = [0.                           , methods['tstop']]
		#if not checkinmethods("imod/current-points" ):methods["imod"]["current-points"] = [-1.*methods["neuron"]["Iapp"], -1.*methods["imod"]["scale"]*methods["neuron"]["Iapp"] ]
		#print " > Time Points                  : ",methods["imod"]["time-points"   ]
		#print " > Current Points               : ",methods["imod"]["current-points"]
		#imodAll = []
		
		#for idx, n in enumerate(neurons):
			#imodt, imodw = h.Vector(), h.Vector()
			#imodt.from_python(methods["imod"]["time-points"     ])
			#imodw.from_python(methods["imod"]["current-points"  ])
			#imodw.play(n.innp._ref_mean,imodt,1)
			#imodAll.append( (imodt,imodw) )
			##++++
		#print "==================================\n"
	
	print "=================================="
	print "===           RUN              ==="
	npc = h.ParallelContext()
	if checkinmethods("ncon"):
		mindel = np.array([ x[0].delay for x in connections ] )
		mindel = np.min(mindel)
		if mindel > h.dt*2:
			if type(methods['corefunc']) is int:
				npc.nthread(methods['corefunc'])
				sys.stderr.write( " > Setup                            : %g core\n"%(methods['corefunc']) )
			else:
				#### Setup parallel context if there are delays
				if not os.path.exists("/etc/beowulf") and os.path.exists("/sysini/bin/busybox"):
					#I'm not on head node. I can use all cores (^-^)
					methods['corefunc'] = methods['corefunc'][2]
					npc.nthread(methods['corefunc'])
					sys.stderr.write( " > Setup                        : %g core\n"%(methods['corefunc']) )
				elif os.path.exists("/etc/beowulf"):
					#I'm on head node. I grub only half (*_*)
					methods['corefunc'] = methods['corefunc'][1]
					npc.nthread(methods['corefunc'])
					sys.stderr.write( " > Setup                        : %g core\n"%(methods['corefunc']) )
				else:
					#I'm on Desktop (-.-)
					methods['corefunc'] = methods['corefunc'][0]
					npc.nthread(methods['corefunc'])
					sys.stderr.write( " > Setup                        : %g cores\n"%(methods['corefunc']) )
		else:
			#I'm on Desktop (v-v)
			methods['corefunc'] = methods['corefunc'][0]
			npc.nthread(methods['corefunc'])
			sys.stderr.write( " > Setup                        : %g cores\n"%(methods['corefunc']) )

	if checkinmethods("cvode"):
		cvode = h.CVode()
		cvode.active(1)
		print " > CVODE                    : ON"
		
		
	h.finitialize()
	h.fcurrent()
	h.frecord_init()
		
	while h.t < methods['tstop']:h.fadvance()

	print "==================================\n"
		


	print "=================================="
	print "===          Analysis          ==="
	print "==================================\n"
	
	t = np.array(t)
	if checkinmethods('gui'):
		plt.rc('text', usetex = True )
		plt.rc('font', family = 'serif')
		plt.rc('svg', fonttype = 'none')
		mainfig = plt.figure(1)
		if checkinmethods('traceView'):
			cid = mainfig.canvas.mpl_connect('button_press_event', onclick1)
		nplot = 411
		if checkinmethods('simvar')      : nplot += 100
		if checkinmethods('spectrogram') : nplot += 100
		p = plt.subplot(nplot)
			
		tprin=np.array(t)
		tprin = tprin[ np.where( tprin < methods['tv'][1] ) ]
		if methods['tv'][0] <= 0.:
			tproc = 0
		else:
			tproc = tprin[ np.where( tprin < methods['tv'][0] ) ]
			tproc = tproc.shape[0]
		tprin = tprin[tproc:]
		vindex = (methods["ncell"]-1)/2
		vtrace, = plt.plot(tprin,np.array(neurons[vindex].volt)[tproc:tprin.size+tproc],"k")
		plt.ylim(ymax=40.)
		mainfig.canvas.mpl_connect('key_press_event',neuronsoverview)
		plt.ylabel("Voltage (mV)", fontsize=16)
		if methods["external"]:
			ex0 = methods["external"]['start']
			ex1 = methods["external"]['interval']
			for ex2 in xrange(methods["external"]['count']):
				plt.plot([ex0+ex1*ex2,ex0+ex1*ex2],[0,30],"r--")
		plt.subplot(nplot+1,sharex=p)
		nurch = np.arange(1,methods["ncell"]+1,1)
		if checkinmethods('sortbysk'):
			if methods['sortbysk'] == 'I':
				nindex = [ (-neurons[i].innp.mean,i) for i in xrange(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in xrange(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'G':
				nindex = [ (-neurons[i].gsynscale,i) for i in xrange(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in xrange(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'NC':
				nindex = [ (-neurons[i].concnt,i) for i in xrange(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in xrange(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'GT':
				nindex = [ (-neurons[i].gtotal,i) for i in xrange(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in xrange(methods["ncell"]):
					nurch[nindex[i][1]]=i
			if methods['sortbysk'] == 'ST':
				nindex = [ (neurons[i].tsynscale,i) for i in xrange(methods["ncell"])]
				nindex.sort()
				#print nindex
				for i in xrange(methods["ncell"]):
					nurch[nindex[i][1]]=i
				
			#print nurch
	pmean = 0.
	pcnt  = 0


	meancur = np.zeros(t.size)
	if checkinmethods('spectrogram'):
		populationcurrent = np.zeros(t.size)
	spbins  = np.zeros( int(np.floor(methods['tstop']))+1 )
	specX	= np.arange(spbins.size, dtype=float)
	specX	*= 1000.0/methods['tstop']
	#pnum	= specX.size/2
	pnum 	= int(200.*methods['tstop']/1000.0)
	specX	= specX[:pnum]
	if checkinmethods("nrnFFT"):
		specN	= np.zeros(pnum)
#	specV	= np.zeros(t.size())
	if 10 < methods["nrnISI"] <= 3000:
		isi		= np.zeros(methods["nrnISI"])
	if checkinmethods('coreindex'):
		coreindex = [0.0, 0.0]

	if checkinmethods('gui'):
		xrast = np.array([])
		yrast = np.array([])
	if checkinmethods('jitter-rec'):
		jallspikes = np.zeros(int((methods['tstop']-methods['cliptrn'])/methods['timestep'])\
		                       if checkinmethods('cliptrn') else \
		                      int(methods['tstop']/methods['timestep']) )  
	for (idx,n) in map(None,xrange(methods["ncell"]),neurons):
		n.spks = np.array(n.spks)
		if checkinmethods('gui'):
			spk = n.spks[ np.where (n.spks < methods['tv'][1]) ]
			if methods['tv'][0] > 0:
				spk = spk[ np.where (spk > methods['tv'][0]) ]
			if not methods['cliprst']:
				xrast = np.append(xrast,spk)
				yrast = np.append(yrast,np.repeat(nurch[idx],spk.size))
			elif idx%methods['cliprst'] == 0:
				xrast = np.append(xrast,spk)
				yrast = np.append(yrast,np.repeat(nurch[idx],spk.size))
		
		if checkinmethods('cliptrn'):
			fstidx = np.where(n.spks > methods['cliptrn'] )[0]
			if len(fstidx) < 1:
				aisi = None
			else:
				fstidx = fstidx[0]
				aisi = n.spks[fstidx+1:] - n.spks[fstidx:-1]
		else:
			aisi = n.spks[1:] - n.spks[:-1]
		if checkinmethods('coreindex'):
			coreindex[0] += np.sum((aisi[1:] - aisi[:-1])/aisi[:-1])
			coreindex[1] += aisi.size - 1
		if 10 < methods["nrnISI"] <= 3000:
			for i in aisi[ np.where(aisi < methods["nrnISI"]) ]:
				isi[ int(np.floor(i)) ] += 1.0
		if not aisi is None:
			pmean += np.sum(aisi)
			pcnt  += aisi.shape[0]
		if checkinmethods('gui'):
			if methods['tracetail'] == 'total current' or methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'TI' or methods['tracetail'] == 'MTI':
				meancur += np.array(n.isyni.x) + np.array(n.inoise.x)
			elif methods['tracetail'] == 'total synaptic current' or methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'TSI' or methods['tracetail'] == 'MTSI':
				meancur += np.array(n.isyni.x)
			elif methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'TG' or methods['tracetail'] == 'MTG':
				meancur += np.array(n.isyng.x)
		if checkinmethods('spectrogram'):
			populationcurrent += np.array(n.isyni.x) + np.array(n.inoise.x)
			
		spn	= np.zeros(spbins.size)
		for sp in n.spks:
			spbins[ int( np.floor(sp) ) ] +=1
			spn[ int( np.floor(sp) ) ] +=1
		
			if checkinmethods('jitter-rec'):
				jps = int( (sp - methods['cliptrn'])/methods["timestep"] ) if checkinmethods('cliptrn') else int( sp/methods["timestep"] )
				jallspikes[jps] += 1
				
				
		if checkinmethods('cliptrn'):
			spn = spn[methods['cliptrn']:]
		if checkinmethods("nrnFFT"):
			fft = np.abs( np.fft.fft(spn)*1.0/methods['tstop'] )**2
			specN += fft[:pnum]
	
	methods["nrnPmean"] = None if pcnt < 1 else float(pmean)/float(pcnt)
	
	if checkinmethods('jitter-rec') and checkinmethods('cliptrn'):
		jallspikes = jallspikes[ np.where( jallspikes > methods['cliptrn'] ) ]
		
	if checkinmethods('gui'):
		plt.plot(xrast,yrast,"k.",mew=0.75,ms=5)#,ms=10)
		
		if methods['fullrast']	: plt.ylim(ymin=0,ymax=methods["ncell"])
		else			: plt.ylim(ymin=0)
	if checkinmethods("nrnFFT"):
		specN /= float(methods["ncell"])#	specV /= float(methods["ncell"])
		methods["nrnFFT-results"] = { 'sectrum':specN[:pnum], 'freq':specX }
	
	if checkinmethods('cliptrn'):
		spbins = spbins[methods['cliptrn']:]
	
	if checkinmethods('popfr'):
		popfr = np.mean(spbins)
		print "=================================="
		print "===       MEAN FIRING RATE     ==="
		print "  > MFR =           ",popfr
		print "==================================\n"
		methods['popfr-results'] = popfr

	if checkinmethods("netFFT") or checkinmethods("nrnFFT"):
		print "=================================="
		print "===            FFT             ==="
		print "==================================\n"
		fft = np.abs( np.fft.fft(spbins)*1.0/methods['tstop'] )**2
		methods["netFFT-results"] = { 'sectrum':fft[:pnum], 'freq':specX }

	##EN
	#probscale = np.zeros(methods["ncell"] + 1)
	#probscale[0] = 1./float(methods["ncell"] + 1)
	#for x in range(1,methods["ncell"] + 1):
		#probscale[x] = probscale[0]*probscale[x-1]
	#pspbin = np.array([ probscale[int(x)] for x in  spbins] )
	#en = np.sum( (-1)*pspbin*np.log(pspbin) )
	
	if checkinmethods('coreindex'):
		coreindex = coreindex[0]/coreindex[1]
		print coreindex, 1./(1.+ abs(coreindex))
		sys.exit(0)
		#with open("coreindex.csv","w") as fd:
			#for i in coreindex: fd.write("%g\n"%i)
		#coreindex = np.corrcoef(coreindex[:-1],y=coreindex[1:])[0][1]
	
	#external stimulation index
	if checkinmethods('external') and checkinmethods('extprop'):
		print "=================================="
		print "===      Spike Probability     ==="
		spprop = 0
		for etx in xrange(methods['external']['count']):
			lidx = int( np.floor(methods['external']['start']+methods['external']['interval']*etx) )
			ridx = int( np.floor(lidx + methods['external']['interval']*methods['extprop']) )
			spprop += float( np.sum(spbins[lidx:ridx]) )
		spprop /= methods['external']['count']*methods["ncell"]
		methods['extprop-results'] = spprop
		print " > Spike group probability      :",spprop
		print "==================================\n"
			

	if checkinmethods("peakDetec") or checkinmethods("R2"):
		print "=================================="
		print "===         Peak Detector      ==="
		print "==================================\n"
		kernel = np.arange(-methods["gkernel"][1],methods["gkernel"][1],1.)
		kernel = np.exp(kernel**2/methods["gkernel"][0]/(-methods["gkernel"][0]))
		module = np.convolve(spbins,kernel)
		module = module[kernel.size/2:1-kernel.size/2]
		#spbinmax = (np.diff(np.sign(np.diff(module))) < 0).nonzero()[0] + 1
		#spbinmin = (np.diff(np.sign(np.diff(module))) > 0).nonzero()[0] + 1
		spbinmark = []
		for idx in (np.diff(np.sign(np.diff(module))) < 0).nonzero()[0] + 1:
			spbinmark.append([idx,1])
		for idx in (np.diff(np.sign(np.diff(module))) > 0).nonzero()[0] + 1:
			spbinmark.append([idx,-1])
		peakmark  = []
		spc,ccnt = 0.,0.
		if len(spbinmark) > 2:
			spbinmark.sort()
			spbinmark = np.array(spbinmark)
			for mx in np.where( spbinmark[:,1] > 0 )[0]:
				if mx <= 2 or mx >= (spbinmark.shape[0]/2 -2):continue
				if spbinmark[mx-1][1] > 0 or spbinmark[mx+1][1] > 0 or spbinmark[mx][1] < 0:continue
				peakmark.append(spbinmark[mx])
				ccnt += 1
				spc += np.sum(spbins[spbinmark[mx-1][0]:spbinmark[mx+1][0]])
		else:
			spbinmark = None
		if ccnt > 0:
			spc /= ccnt
		
	if checkinmethods('jitter-rec'):
		print "=================================="
		print "===       Jitter Detector      ==="
		print "==================================\n"
		jkernel = np.arange(-methods["gkernel"][1],methods["gkernel"][1],methods['timestep'])
		jkernel = np.exp(jkernel**2/methods["gkernel"][0]/(-methods["gkernel"][0]))
		jmodule = np.convolve(jallspikes,jkernel)
		jmodule = jmodule[jkernel.size/2:1-jkernel.size/2]
		jpeaks = []
		#for idx in (np.diff(np.sign(np.diff(jmodule))) < 0).nonzero()[0] + 1:
			#jpeaks.append(float(idx))
		#for il,ic,ir in zip(jpeaks[:-2],jpeaks[1:-1],jpeaks[2:]):
			#for ik in jmodule[(il+ic)/2:ic]:
			#???*methods['timestep']
		

	##R2
	##Per
	if checkinmethods("R2"):
		methods["R2-results"] = {}
		if ccnt > 0:
			methods["R2-results"]['spc'] = spc
		else:
			methods["R2-results"]['spc'] = None
		print "=================================="
		print "===             R2             ==="
		X,Y,Rcnt,netpermean,netpercnt=0.,0.,0.,0.0,0.0
		phydist  = []
		if not spbinmark is None:
			for mx in np.where( spbinmark[:,1] > 0 )[0]:
				if mx >= (spbinmark.shape[0]/2 - 3):continue
				if spbinmark[mx+1][1] > 0 or spbinmark[mx+2][1] < 0 or spbinmark[mx][1] < 0:continue
				Pnet = float(spbinmark[mx+2][0] - spbinmark[mx][0])
				netpermean += Pnet
				netpercnt  += 1.
				for n,i in map(None,spbins[spbinmark[mx][0]:spbinmark[mx+2][0]],xrange(spbinmark[mx+2][0] - spbinmark[mx][0])):
					phyX = np.cos(np.pi*2.*float(i)/Pnet)
					phyY = np.sin(np.pi*2.*float(i)/Pnet)
					X += n*phyX
					Y += n*phyY
					Rcnt += n
					if methods['sycleprop']:
						#phydist.append( (360.*np.arctan2(phyY,phyX)/2/np.pi,n) )
						#phydist.append( (np.arctan2(phyY,phyX),n) )
						phydist.append( (np.pi*2.*float(i)/Pnet,n) )
		if Rcnt > 0.:
			R2 = (X/Rcnt)**2+(Y/Rcnt)**2
			methods["R2-results"]["R2"] = R2
		else:
			methods["R2-results"]["R2"] = None
		if netpercnt > 1.:
			netpermean /= ( netpercnt - 1)
			methods["R2-results"]["netPmean"] = netpermean
		else:
			methods["R2-results"]["netPmean"] = None
		print "  > R2       =           ",methods["R2-results"]["R2"]
		print "  > SPC      =           ",methods["R2-results"]['spc']
		print "  > netPmean =           ",methods["R2-results"]["netPmean"]
		print "==================================\n"
		if checkinmethods('sycleprop'):
			phydist = np.array(phydist)
			phydist[:,1] /= np.sum(phydist[:,1])
			phyhist,phyhistbins = np.histogram(phydist[:,0], bins=37, weights=phydist[:,1],range=(-np.pi/36,2.*np.pi+np.pi/36))
			methods['sycleprop-results'] = { 'histogram':phyhist, 'bins-bounders':phyhistbins }
			
		
	if 10 < methods["netISI"] < 3000:
		print "=================================="
		print "===          NET ISI           ==="
		print "==================================\n"
		netisi	= np.zeros(methods["netISI"])
		lock = threading.RLock()
		
		def calcnetisi(ns):
			global netisi, lock
			scans	= np.zeros(methods["ncell"],dtype=int)
			localnetisi = np.zeros(methods["netISI"])
			for n in ns:
				for sp in n.spks:
					for (idx,m) in map(None,xrange(methods["ncell"]),neurons):
						if m.spks.size < 2 : continue
						while m.spks[scans[idx]] <= sp and scans[idx] < m.spks.size - 1 : scans[idx] += 1
						if m.spks[scans[idx]] <= sp : continue
						if m == n and m.spks[scans[idx]] - sp < 1e-6 : continue
						aisi = m.spks[scans[idx]] - sp
						if int(aisi) >= methods["netISI"] : continue
						localnetisi[ int(aisi) ] += 1
			with lock:
				netisi += localnetisi
		pids = [ threading.Thread(target=calcnetisi, args=(neurons[x::methods['corefunc']],)) for x in xrange(methods['corefunc']) ]
		for pidx in pids:
			pidx.start()
			#print pidx, "starts"
		for pidx in pids:
			pidx.join()
			#print pidx,"finishs"
		methods['netISI-results'] = netisi
		
	if checkinmethods("T&S") or checkinmethods('lastspktrg'):
		print "=================================="
		print "===           T & S            ==="
		print "==================================\n"
		allspikes,activeneurons = [],0.
		for n in neurons:
			allspikes += list(n.spks)
			if n.spks.size !=0 :activeneurons += 1.
		allspikes.sort()
		allspikes = np.array(allspikes)
		TaSisi = allspikes[1:]-allspikes[:-1]
		if checkinmethods('lastspktrg'):
			lastspktrg = int( np.mean(allspikes) > methods['tstop']/4. )
			methods['lastspktrg-results'] = lastspktrg
		
		del allspikes
		if checkinmethods("T&S"):
			if bool(lastspktrg):
				mean1TaSisi = np.mean(TaSisi)
				TaSindex	= (np.sqrt(np.mean(TaSisi**2) - mean1TaSisi**2)/mean1TaSisi - 1.)/np.sqrt(activeneurons) 
				methods['T&S-results'] = TaSindex
			else:
				methods['T&S-results'] = None
		
	if checkinmethods("Delay-stat"):
		print "=================================="
		print "===     Delays distribution    ==="
		delays = np.array([ x[0].delay for x in connections])
		mdly, sdly,mxdly,Mxdly = np.mean(delays), np.std(delays), np.min(delays), np.max(delays)
		if not type(methods["Delay-stat"]) is tuple:
			methods["Delay-stat"] = (0., 15., 500)
		dlyhist,dlybins = np.histogram(delays, bins=methods["Delay-stat"][2], normed=True, range=methods["Delay-stat"][0:2] )
		dlyhist /= np.sum(dlyhist)
		print "  > Delays mean  =           ",mdly
		print "  > Delays stdev =           ",sdly
		print "  > Delays min   =           ",mxdly
		print "  > Delays max   =           ",Mxdly
		methods['Delay-stat-results'] = {
			'mean': mdly, 'stdev': sdly, 'min': mxdly, 'max': Mxdly, 'histogram':dlyhist, 'bins-bounders':dlybins
		}
		print "==================================\n"
		
	if checkinmethods('Gtot-dist') :
		print "=================================="
		print "===   G-total  distribution    ==="
		#gsk = [ n.gsynscale for n in neurons ]
		gsk = [ n.gtotal for n in neurons ]
		mgto, sgto,mxgto,Mxgto = np.mean(gsk), np.std(gsk), np.min(gsk), np.max(gsk)
		gskhist,gskbins = np.histogram(gsk, bins=methods["ncell"]/25, normed=True, range=[0,Mxgto])#/10)#, normed=True)
		gskhist /= np.sum(gskhist)
		print "  > Total Syn. Cond mean  =  ",mgto*1e5
		print "  > Total Syn. Cond stdev =  ",sgto*1e5
		print "  > Total Syn. Cond min   =  ",mxgto*1e5
		print "  > Total Syn. Cond max   =  ",Mxgto*1e5
		methods['Gtot-dist-results'] = { 
			'mean': mgto, 'stdev': sgto, 'min': mxgto, 'max': Mxgto,
			'histogram':gskhist, 'bins-bounders':gskbins 
		}
		del gsk
		print "==================================\n"
		
	if checkinmethods('Gtot-stat'):
		print "=================================="
		print "===     G-total Statistics     ==="
		agtot = np.array([ n.gtotal/n.concnt for n in neurons ])
		#DB>>
		#print [n.gtotal for n in neurons ]
		#print [n.concnt for n in neurons ]
		#exit(0)
		#<<DB
		mgtot = np.mean(agtot)
		sgtot = np.std(agtot)
		print "  > mean   gtotal =           ",mgtot
		print "  > stderr gtotal =           ",sgtot
		print "  > CV     gtotal =           ",sgtot/mgtot
		print "==================================\n"
		methods['Gtot-stat-results'] = { 'mean':mgtot, 'stdev':sgtot, 'CV':sgtot/mgtot }

	
	if checkinmethods('2cintercon'):
		print "=================================="
		print "===  2 clusters connectivity   ==="
		tims = methods['tstop']*3./4.
		if pcnt != 0:
			halfpnet = pmean/pcnt/2.
			clslst = []
			print "  >  Searching for clusters       "
			rarr = np.array(neurons[0].spks)
			tims = rarr[ np.where( rarr > tims ) ]
			del rarr
			tims = tims[0] + halfpnet/2.
			print "  >  Time to search              :",tims
			for idx, n in enumerate(neurons):
				getlest = np.array(n.spks)
				getlest = getlest[ np.where( getlest>tims )]
				if getlest.shape[0] < 1: continue
				if getlest[0] > tims+halfpnet:
					clslst.append(True)
				else:
					clslst.append(False)
			print "  >  Searching for connectivity index"
			WithinA, WithinB, CrossAB, CrossBA = 0, 0, 0, 0
			countA, countB = 0, 0
			fullstat = checkinmethods('2clrs-stat')
			if fullstat:
				within,cross = np.zeros(methods['ncell']),np.zeros(methods['ncell'])
			for idx, cnt in enumerate(OUTList):
				if clslst[idx]:
					#Cluster A
					countA += 1
					for c in cnt:
						if clslst[c]:
							WithinA += 1
							if fullstat : within[idx] += 1
						else:
							CrossAB += 1
							if fullstat : cross[idx] += 1
				else:
					#cluster B
					countB += 1
					for c in cnt:
						if clslst[c]:
							CrossBA += 1
							if fullstat : cross[idx] += 1
						else:
							WithinB += 1
							if fullstat : within[idx] += 1
			print "  >  Cells in the Cluster A      :",countA
			print "  >  Cells in the Cluster B      :",countB
			print "  >  Within Cluster A            :",WithinA
			print "  >  Within Cluster B            :",WithinB
			print "  >  From A to B                 :",CrossAB
			print "  >  From A to B                 :",CrossBA
			print "  >  Total Within Both Clusters  :",WithinA + WithinB
			print "  >  Total Between Both Clusters :",CrossAB + CrossBA
			print "  >  Ratio Between to Within     :",float(CrossAB + CrossBA)/float(WithinA + WithinB)
			if fullstat:
				print "  >  FUUL STATISTICS "
				#print "  >  ",
				#for idx,(win,btwn) in enumerate(zip(within,cross)):
					#print "{}:{}".format(win,btwn),
					#if not bool((idx+1)%6): print "\n  >  ",
				#print "  >  RATIOS "
				#print "  >  ",
				#for idx,(win,btwn) in enumerate(zip(within,cross)):
					#print "{}".format(btwn/win),
					#if not bool((idx+1)%6): print "\n  >  ",
				#print "  >  "
				withinA = within[ np.where( np.array(clslst) ) ]
				crossA = cross[ np.where( np.array(clslst) ) ]
				print "  >  Cluster A within mean,stdev :",np.mean(withinA), np.std(withinA)
				print "  >  Cluster A to B mean,stdev   :",np.mean(crossA), np.std(crossA)
				withinB = within[ np.where(  (1-1*np.array(clslst).astype(int)).astype(bool) )]
				crossB = cross[ np.where(  (1-1*np.array(clslst).astype(int)).astype(bool) )]
				print "  >  Cluster B within mean,stdev :",np.mean(withinB), np.std(withinB)
				print "  >  Cluster B to A mean,stdev   :",np.mean(crossB), np.std(crossB)
			methods['2cintercon-results'] = {
				"cells-in-A":countA, "cells-in-B":countB,
				'connections-in-A':WithinA, 'connections-in-B':WithinB,
				"connections-A2B":CrossAB, "connections-B2A":CrossBA,
				'total in A&B':WithinA + WithinB,
				'total between A&B':CrossAB + CrossBA,
				'total ratio between/in':float(CrossAB + CrossBA)/float(WithinA + WithinB)
			}
		else:
			print "  >  Pnet isn't defined...,       "
			methods['2cintercon-results'] = None
		print "==================================\n"
	
	if checkinmethods('get-steadystate'):
		if type(methods['get-steadystate']) is str:
			ssthr=+30.
			ssfilename = methods['get-steadystate']
		elif type(methods['get-steadystate']) is float or type(methods['get-steadystate']) is int:
			ssthr = float(methods['get-steadystate'])
			if type(methods["neuron"]["Vinit"]) is str:
				ssfilename =  methods["neuron"]["Vinit"]+"-ss.dat"
			else:
				ssfilename = 'get-steadystate.dat'
		else:
			ssthr=+30.
			ssfilename = 'get-steadystate.dat'
		print "=================================="
		print "===     Write Steady State     ==="
		print "  >  Threshold                   :",ssthr
		print "  >  Output File                 :",ssfilename
		ssvec = np.array(neurons[0].volt)
		ssmsk = np.where( (t>methods['tstop']*4./5.)*(ssvec >= ssthr) )[0]
		if ssmsk.shape[0] == 0:
			print "Error Cannot Get Voltage above Threshold!"
			with open(ssfilename,"w") as fd:
				fd.write("None")
				for n in neurons[1:]:
					fd.write(" None")
				fd.write("\n")
		else:
			ssmsk = int(ssmsk[0])
			with open(ssfilename,"w") as fd:
				fd.write("%g"%ssvec[ssmsk])
				for n in neurons[1:]:
					fd.write(" %g"%(np.array(n.volt)[ssmsk]))
				fd.write("\n")
		print "==================================\n"
	#EN
	#p.set_title("Mean individual Period = %g, Sychrony(Entropy) = %g(%g)"%(pmean/pcnt,1./(1.+en),en))
	
	##R2
	if checkinmethods('gui'):
		title = "Mean individual Period = %s"%("NONE" if pcnt == 0 else "%g"%(pmean/pcnt))
		if checkinmethods('popfr'):
			title += 'Mean FR =%g'%popfr
		if checkinmethods("R2"):
			if Rcnt > 0 :
				title += r", $R^2$ = %g, Mean network Period = %g, Spike per cycle = %g"%(R2,netpermean,spc)
			else:
				title += ", *Fail to estimate network period*"
		elif checkinmethods("peakDetec"):
			title += ", Spike per cycle = %g. "%(spc)
		if checkinmethods('T&S'):
			title += ", TaS = %g"%TaSindex
		if checkinmethods('lastspktrg'):
			title += ", LST = %g"%lastspktrg
		p.set_title(title)

		
		plt.subplot(nplot+2,sharex=p)
		if checkinmethods('cliptrn'):
			nppoints = np.arange(methods['tv'][0]+methods['cliptrn'],methods['tv'][1],1.0)
			plt.bar(nppoints,spbins[:methods['tv'][1]-methods['cliptrn']],0.5,color="k")
			hight = spbins[:methods['tv'][1]-methods['cliptrn']].max()
			if (checkinmethods("peakDetec") or checkinmethods("R2")) and not spbinmark is None :
				for mark in spbinmark:
					if mark[0]+methods['cliptrn'] < methods['tv'][0] or mark[0]+methods['cliptrn'] > methods['tv'][1]: continue
					if mark[1] > 0:
						plt.plot([mark[0]+methods['cliptrn'],mark[0]+methods['cliptrn']],[0,hight],"r--")
					else:
						plt.plot([mark[0]+methods['cliptrn'],mark[0]+methods['cliptrn']],[0,hight],"b--")
		else:
			nppoints = np.arange(methods['tv'][0],methods['tv'][1],1.0)
			plt.bar(nppoints,spbins[int(methods['tv'][0]):int(methods['tv'][1])],0.5,color="k")
			hight = spbins[int(methods['tv'][0]):int(methods['tv'][1])].max()
			if (checkinmethods("peakDetec") or checkinmethods("R2")) and not spbinmark is None :
				for mark in spbinmark:
					if mark[0] < methods['tv'][0] or mark[0] > methods['tv'][1]: continue
					if mark[1] > 0:
						plt.plot([mark[0],mark[0]],[0,hight],"r--")
					else:
						plt.plot([mark[0],mark[0]],[0,hight],"b--")
#			plt.plot(nppoints,module[methods['tv'][0]:methods['tv'][1]]/np.sum(kernel),"k--")
#			plt.plot(nppoints,module[methods['tv'][0]:methods['tv'][1]],"k--")
		plt.ylabel("Rate (ms$^{-1}$)", fontsize=16)

	if checkinmethods('gui'):
		if methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'MTI':
			meancur = meancur / float(-methods["ncell"])
		elif methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI':
			meancur = meancur / float(-methods["ncell"])
		elif methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'MTG':
			meancur = meancur / float(methods["ncell"])
	
	if checkinmethods('gui'):
		
		plt.subplot(nplot+3,sharex=p)
			
		if methods['tracetail'] == 'total current' or methods['tracetail'] == 'TI' or methods['tracetail'] == 'mean total current' or methods['tracetail'] == 'MTI'\
		  or methods['tracetail'] == 'total synaptic current' or methods['tracetail'] == 'TSI' or methods['tracetail'] == 'mean total synaptic current' or methods['tracetail'] == 'MTSI':
			plt.ylabel("Current (nA)", fontsize=16)
			plt.plot(tprin,-meancur[tproc:tprin.size+tproc]*1e5)
		elif methods['tracetail'] == 'total conductance' or methods['tracetail'] == 'mean total conductance' or methods['tracetail'] == 'TG' or methods['tracetail'] == 'MTG':
			plt.ylabel("Total Conductance (nS)", fontsize=16)
			plt.plot(tprin,meancur[tproc:tprin.size+tproc]*1e5)
		elif methods['tracetail'] == 'firing rate' and ( methods["peakDetec"] or methods["R2"] ):
			plt.ylabel("Firing Rate (ms$^{-1}$)", fontsize=16)
			plt.plot(nppoints,module[methods['tv'][0]:methods['tv'][1]]/np.sum(kernel),"k--")
			hight = np.max(module[methods['tv'][0]:methods['tv'][1]]/np.sum(kernel))
			if not spbinmark is None :
				for mark in spbinmark:
					if mark[0] < methods['tv'][0] or mark[0] > methods['tv'][1]: continue
					if mark[1] > 0:
						plt.plot([mark[0],mark[0]],[0,hight],"k--")
		elif methods['tracetail'] == 'conductance':
			plt.ylabel("Conductance (nS)", fontsize=16)
			xvcrv, = plt.plot(tprin,np.array(neurons[vindex].isyng)[tproc:tprin.size+tproc]*1e5,'k-',lw=2)
		elif methods['tracetail'] == 'current':
			plt.ylabel(r"Current ($\mu$A)", fontsize=16)
			xvcrv, = plt.plot(tprin,np.array(neurons[vindex].isyni)[tproc:tproc+tprin.size]*1e5,'k-',lw=2)
		if checkinmethods('simvar'):
			simvarrec = np.array(simvarrec)
			plt.subplot(nplot+4,sharex=p)
			#plt.ylabel(methods['simvar']['var'], fontsize=16)
			plt.ylim(min([methods['simvar']["a0"],methods['simvar']["a1"]]),max([methods['simvar']["a0"],methods['simvar']["a1"]]))
			plt.plot(tprin,simvarrec[tproc:tprin.size+tproc])
		if checkinmethods('spectrogram'):
			plt.subplot(nplot+(5 if checkinmethods('simvar') else 4),sharex=p)
			populationcurrent
			#NFFT = 131072       # the length of the windowing segments
			NFFT = 65535       # the length of the windowing segments
			Fs = int(1000.0/methods['timestep'])  # the sampling frequency
			from scipy.signal import spectrogram
			f, tf, Sxx = spectrogram(populationcurrent, fs=Fs, nperseg=NFFT,noverlap=NFFT*1020/1024,window='hanning')
			Sxx = Sxx[np.where(f<100),:][0]
			f   = f[np.where(f<100)]
			Sxx = Sxx[np.where(f>20),:][0]
			f   = f[np.where(f>20)]
			print "T SHAPE",tf.shape
			print "T      ",tf
			print "F SHAPE",f.shape
			print "F      ",f
			print "S SHAPE",Sxx.shape
			print "s      ",Sxx
			plt.pcolormesh(tf*1e3, f, Sxx)

		plt.xlabel("time (ms)", fontsize=16)


	
	if (checkinmethods("netFFT") or checkinmethods("nrnFFT")) and checkinmethods('gui'):
		plt.figure(2)
		if checkinmethods("netFFT") and checkinmethods("nrnFFT"):
			pl=plt.subplot(211)
		elif checkinmethods("netFFT"):
			pl=plt.subplot(111)
		if checkinmethods("netFFT"):
			plt.title("Network spectrum")
			plt.bar(specX[1:],fft[1:pnum],0.75,color="k",edgecolor="k")
		if checkinmethods("netFFT") and checkinmethods("nrnFFT"):
			plt.subplot(212,sharex=pl)
		elif checkinmethods("nrnFFT"):
			plt.subplot(111)
		if checkinmethods("nrnFFT"):
			plt.title("Neuronal spectrum")
			plt.bar(specX[1:],specN[1:],0.75,color="k",edgecolor="k")

	#plt.subplot(313,sharex=p)
	#specX =np.arange(0.0,methods['tstop']+h.dt,h.dt)
	#specX *= 1000.0/methods['tstop']/h.dt
	#pnum = specX.size/2
	#plt.title("Voltage spectrum")
	#plt.plot(specX[1:pnum],specV[1:pnum])
	#plt.xlim(0,200)
	
	if 10 < methods["netISI"] <= 3000 and sum(netisi) > 0: netisi /= sum(netisi)
	if 10 < methods["nrnISI"] <= 3000 and sum(isi) > 0: isi /= sum(isi)
	if (10 < methods["netISI"] <= 3000 or 10 < methods["nrnISI"] <= 3000) and methods['gui']:
		plt.figure(3)
		if 10 < methods["netISI"] <= 3000 and 10 < methods["nrnISI"] <= 3000:
			pl=plt.subplot(211)
		elif 10 < methods["netISI"] <= 3000 :
			plt.subplot(111)
		if 10 < methods["netISI"] <= 3000: 
			plt.title("Network ISI")
			plt.ylabel("Normalized number of interspike intervals", fontsize=16)
			plt.bar(np.arange(methods["netISI"]),netisi,0.75)
		if 10 < methods["netISI"] <= 3000 and 10 < methods["nrnISI"] <= 3000:
			plt.subplot(212)#,sharex=pl)
		elif 10 < methods["nrnISI"] <= 3000:
			plt.subplot(111)
		if 10 < methods["nrnISI"] <= 3000:
			plt.ylabel("Normalized number of interspike intervals", fontsize=16)
			plt.title("Neuronal ISI")
			plt.bar(np.arange(methods["nrnISI"]),isi,0.75,color='k')
			plt.xlim(0,methods["nrnISI"])
			plt.xlabel("Interspike intervals (ms)", fontsize=16)
	
	if checkinmethods('traceView') and checkinmethods('gui'):
		zooly = plt.figure(4)
		zooly.canvas.mpl_connect('button_press_event', zoolyclickevent)
		zooly.canvas.mpl_connect('key_press_event', zoolykeyevent)
		moddy = plt.figure(5)
		faxi = plt.subplot2grid((6,10),(0,0),colspan=4,rowspan=2)
		faxi.set_ylabel("Presynaptic spikes")
		vaxi = plt.subplot2grid((6,10),(2,0),colspan=4,sharex=faxi)
		vaxi.set_ylabel("V[mV]")
		uaxi = plt.subplot2grid((6,10),(3,0),colspan=4,sharex=faxi)
		uaxi.set_ylabel(methods['traceView'])
		gaxi = plt.subplot2grid((6,10),(4,0),colspan=4,sharex=faxi)
		gaxi.set_ylabel(r"$g_{syn} [xS]$")
		iaxi = plt.subplot2grid((6,10),(5,0),colspan=4,sharex=faxi)
		iaxi.set_ylabel(r"$I_{syn} [xA]$")
		#saxi = plt.subplot2grid((6,10),(5,0),colspan=4,sharex=faxi)
		naxi = plt.subplot2grid((6,10),(0,5),colspan=6,rowspan=6)
		naxi.set_ylabel(methods['traceView'])
		naxi.set_xlabel("V[mV]")
		moddy.canvas.mpl_connect('key_press_event', moddykeyevent)# zoolykeyevent)
		moddy.canvas.mpl_connect('button_press_event', moddyclickevent)


	if checkinmethods('GPcurve') and checkinmethods('gui'):
		plt.figure(7)
		f  = np.array([ [n.gsynscale,n.spks.size]       for n in neurons])
		#f = np.sort(f, axis=0)
		plt.plot(f[:,0] ,f[:,1],"k+")
			
	if checkinmethods('sycleprop') and checkinmethods('gui'):
		plt.figure(8)
		polarax = plt.subplot(111, polar=True)
		#bars = polarax.bar(phydist[:,1], phydist[:,0], width=0.25, bottom=0.0)
		#np.histogram(phydist[:,0], bins=180, weights=phydist[:,1])
		#polarax.hist(phydist[:,0], bins=36, weights=phydist[:,1])
		polarax.bar(phyhistbins[:-1],phyhist,width=phyhistbins[1]-phyhistbins[0],bottom=0)
		#DB>>
		plt.figure(9)
		plt.bar(phyhistbins[:-1],phyhist,width=phyhistbins[1]-phyhistbins[0],bottom=0)
		#<<DB
	if checkinmethods('Gtot-dist') and checkinmethods('gui'):
		plt.figure(10)
		plt.bar(gskbins[:-1]*1e5,gskhist,width=(gskbins[1]-gskbins[0])*1e5,color="k")
		#plt.hist(gsk,bins=methods["ncell"]/50)
		plt.title("mena total synaptic conductance={}(ms), stdr total synaptic conductance={}(nS)".format(mgto*1e5,sgto*1e5) )
		plt.ylabel("Probability")
		plt.xlabel("Toaol conductance (nS)")

	if checkinmethods("Delay-stat") and checkinmethods('gui'):
		plt.figure(11)
		plt.bar((dlybins[1:]+dlybins[:-1])/2.,dlyhist,width=dlybins[1]-dlybins[0],color="k")
		plt.title("mena delay={}(ms), stdr delay={}(ms)".format(mdly,sdly))
		plt.ylabel("Probability")
		plt.xlabel("delay(ms)")


	if checkinmethods('git'):
		os.system("git commit -a &")
	if checkinmethods('beep'):
		os.system("beep &")
	if checkinmethods('corelog'):
		def writetree(tree,fd,prefix):
			for name in tree:
				if type(tree[name]) is dict:
					writetree(tree[name],fd,prefix+name+'/')
				else:
					#DB>>
					#print prefix,name,tree[name]
					#<<DB
					fd.write(":{}={}".format(prefix+name,tree[name]))
		with open(methods['corelog']+".simdb","a") as fd:
			now = datetime.now()
			fd.write("SIMDB/TIMESTAMP=(%04d,%02d,%02d,%02d,%02d,%02d)"%(now.year, now.month, now.day, now.hour, now.minute, now.second) )
			writetree(methods,fd,"/")
			fd.write("\n")
	if methods['gui']:
		if methods['gif']:
			plt.savefig(methods['gif'])
		else:
			plt.show()
	if not methods['noexit']:
		sys.exit(0)
