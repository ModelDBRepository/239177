# -*- coding: utf-8 -*-
"""
/***********************************************************************************************************\

 This NEURON + Python scripts associated with paper:                                                        
  Ruben A. Tikidji-Hamburyan, Carmen C. Canavier
  Robust One and Two Cluster Synchrony Mediated by Resonant Interneurons in Sparsely but 
    Strongly Connected Inhibitory and Excitatory/Inhibitory Networks
 
 The network consists of 1120 excitatory (Olufsen et al. 2003), 280 inhibitory HH neuron and 1120 independent 
  external Poisson's drives to E cell. Number of neurons can be change over command line arguments (see below).
 All neurons are sparsely connected by double exponential synapses.
 
 All parameters may be set up by command line arguments listed AFTER script name. The command should be:   
  nrngui -nogui -python network.py [PARAMETERS]                                                            
 A list of available parameters can be printed out by command:                                              
  nrngui -nogui -python pirping-network.py --help
 
 ---
  
 To run simulation for any point in Figure 7B run 
  nrngui -nogui -python pirping-network.py -c=PIR-PING-2.2ms-rnd.conf /Connections/II/gmax-mean=<synaptic conductance in uS>
 See examples below.
 
 ---
 
 To replicate exact Figure 7A1 run:
  nrngui -nogui -python pirping-network.py -f=example-asyn.conf
 
 To regenerate network with parameters as in Figure 7A1 but with random 
 set of connections and Poisson's processes spikes (regenerated stochasticity) run:
  nrngui -nogui -python pirping-network.py -f=PIR-PING-2.2ms-rnd.conf /Connections/II/gmax-mean=1.333521e-5
 
 To replicate exact Figure 7A2 run:
  nrngui -nogui -python pirping-network.py -f=example-ping.conf
 or with regenerated stochasticity :
  nrngui -nogui -python pirping-network.py -f=PIR-PING-2.2ms-rnd.conf /Connections/II/gmax-mean=1.0e-06

 To replicate exact Figure 7A3 run:
  nrngui -nogui -python pirping-network.py -f=example-2clt.conf
 or with regenerated stochasticity :
  nrngui -nogui -python pirping-network.py -f=PIR-PING-2.2ms-rnd.conf /Connections/II/gmax-mean=7.498942e-05

 To replicate exact Figure 7A4 run:
  nrngui -nogui -python pirping-network.py -f=example-sync.conf
 or with regenerated stochasticity :
  nrngui -nogui -python pirping-network.py -f=PIR-PING-2.2ms-rnd.conf /Connections/II/gmax-mean=1.0e-02


 
Copyright: Ruben Tikidji-Hamburyan <rtikid@lsuhsc.edu> <ruben.tikidji.hamburyan@gmail.com> Mar.2017 - May.2017

\************************************************************************************************************/  
"""

import sys, os, itertools, logging
try:
	import cPickle as pkl
except:
	import pickle as pkl

from numpy import *
from numpy import random as rnd
import numpy as np
import scipy as sp
import scipy.fftpack as spfft
import scipy.signal as spsignal


from neuron import h
from simtools import *


def getspikesequence(i,start,stop, f):
	x = [start]
	while x[-1] < stop:
		w = f(0,x[-1])
		while w-x[-1] < 0.02: w = f(i,x[-1])
		x.append(w)
	return array(x)
def equalcodesize(maxsize):
	def getn(n):
		return int( reduce(lambda x,y:x*y,range(n/2+1,n+1),1)/reduce(lambda x,y:x*y,range(1,n/2+1),1) )
	return int( next(  x[0] for x in [ (k,getn(k)) for k in range(2,200,2) ] if x[1] >  maxsize) )
def equalcodegen(size):
	x = [ 1 for _ in xrange(size/2)]+[ 0 for _ in xrange(size/2)]
	def walk(seq):
		idx=0
		while seq[idx] != 0: idx+=1
		if idx == 0: 
			s = seq[:]
			yield []
		else:
			seq[idx-1] = 0
			for pos in xrange(idx, len(seq)):
				s = seq[:]
				s[pos] = 1
				yield s
				for p in walk(s[:pos]):
					if len(p) == 0: continue
					k = p+s[pos:]
					yield k
				
	
	wset =  [x]+[ x for x in walk(x) ]
	return [ "".join( "%d"%x for x in p) for p in wset ]

#DB>> LOGGER
#logging.basicConfig(format='%(asctime)s:%(module)-10s%(lineno)-6d%(levelname)-8s:%(message)s', level=logging.DEBUG)
#logging.basicConfig(format='%(asctime)s:%(name)-35s:%(lineno)-6d:%(levelname)-8s:%(message)s', level=logging.DEBUG)
#logging.basicConfig(filename="err.log",format='%(asctime)s:%(name)-35s:%(lineno)-6d:%(levelname)-8s:%(message)s', level=logging.DEBUG)
#logging.basicConfig(format='%(asctime)s:%(pathname)-35s:%(lineno)-6d:%(levelname)-8s:%(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s:%(pathname)-35s:%(lineno)-6d:%(levelname)-8s:%(message)s', level=logging.INFO)

######################################################################################################################
# Records for paper
#  PIR-PING 3   ms: 61d1d35d9972dd9e91cf3904ab7dcd0e565898bd
#  PIR-PING 1.3 ms: 464d877a64585ea37cdd2fb975cc71b2c0bd025e
#  PIR-PING-II    : 0265d6e1bf61da5ad157e5ef3c3ffd89895eb356

#<<DB
preset = [
	'/Parallel/MPI/RANK = 1',
	'/Parallel/MPI/SIZE = 10',
	]

defaults = [

	# Functions
	("/Function/TrancatedGauss=lambda meanX, stdX, minX = 0.: next(x for x in (meanX + random.randn() * stdX for _ in iter(int, 1)) if x > minX)",	"A function for truncated Gaussian distribution"),

	#  Simulations parameters
	('/Simulation/step = 0.01', 'Simulations time stap in ms'),
	('/Simulation/stop = 500', 'Time to stop simulation (tstop in NEURONN)'),
	('/Simulation/cvode = False', 'Parameters for CVODE'),
	#('/Simulation/Celsius = 20.', 'Temperature in Celsius'),

	#  Parallel options
	('/Parallel/cores/desktop  =  8', 'Number of cores on desctop'),
	('/Parallel/cores/neadnode = 12', 'Number of cores on beowulf head node'),
	('/Parallel/cores/compute  = 32', 'Number of cores on beowulf compute node'),
	('/Parallel/cores/autodetect = True', 'Autodetect beowulf nodes'),
	('/Parallel/cores/number   = @/Parallel/cores/desktop@', 'Default as on desctop, but set something else and autodetect=False to keep a balance'),

	#  General parameters
	("/ATotalCell=1400",		"Total number of cells"),

	#  Excitatory population
	("/Populations/E/n         = int(@/ATotalCell@*0.8)",	"Total number of neurons in excitatory population"),
	("/Populations/E/model     = \'from ECellOlufsen import Py as NeuronE\'",	"Model of individual neuron in excitatory population"),
	("/Populations/E/modelname = \'NeuronE\'", "Name of neuron class for E- population"),
	("/Populations/E/init      = random.randn(@/Populations/E/n@)*40.-60.", "Initial condition for E-population"),
	("/Populations/E/DB>>/vinit= False", "print initial voltage"),
	("/Populations/E/record/v  = \'soma(0.5)._ref_v\'", "We will record voltage from excitatory cells"),
	("/Populations/E/record/ei = \'esyn._ref_i\'", "record excitatory current in excitatory cells"),
	("/Populations/E/record/eg = \'esyn._ref_g\'", "record excitatory conductance in excitatory cells"),
	("/Populations/E/record/ii = \'isyn._ref_i\'", "record inhibitory current in excitatory cells"),
	("/Populations/E/record/ig = \'isyn._ref_g\'", "record inhibitory conductance in excitatory cells"),

	#  Inhibitory population
	("/Populations/I/n         = int(@/ATotalCell@-@/Populations/E/n@)",	"Total number of neurons in inhibitory population"),
	("/Populations/I/model     = \'from HHinh import In as NeuronI\'",	"Model of individual neuron in inhibitory population"),
	("/Populations/I/modelname = \'NeuronI\'", "Name of neuron class for I- population"),
	("/Populations/I/init      = array(random.randn(@/Populations/I/n@)*40.-70.)", "Initial condition for E-population"),
	("/Populations/I/record/v  = \'soma(0.5)._ref_v\'", "We will record voltage from inhibitory cells"),
	("/Populations/I/record/ei = \'esyn._ref_i\'", "record excitatory current in inhibitory cells"),
	("/Populations/I/record/eg = \'esyn._ref_g\'", "record excitatory conductance in inhibitory cells"),
	("/Populations/I/record/ii = \'isyn._ref_i\'", "record inhibitory current in inhibitory cells"),
	("/Populations/I/record/ig = \'isyn._ref_g\'", "record inhibitory conductance in inhibitory cells"),

	#  Stimulus	

	## Mixed source
#	("/Populations/S/n          = equalcodesize(@/Populations/E/n@)", "Total number of spike sources in an external input"),
#	("/Populations/S/n          = 9", "Total number of spike sources in an external input"),
	#("/Populations/S/DistMean   = 2.5*@/Populations/S/n@", "Mean population firing rate scaled up to population size"),
	#("/Populations/S/DistStder  = 1.5*@/Populations/S/n@", "Desired standard deviation of population firing rate scaled up to population size"),
	#("/Populations/S/GammaShape = (@/Populations/S/DistMean@/@/Populations/S/DistStder@)**2", "Shape of gamma distribution for interspikes intervals"),
	#("/Populations/S/GammaScale = @/Populations/S/DistMean@/@/Populations/S/GammaShape@", "Scale for interspike intervals distribution"),
	#("/Populations/S/spikesource= lambda i, tpre: tpre+random.gamma(@/Populations/S/GammaShape@, scale=@/Populations/S/GammaScale@)",
		#"Function which generate next spike on i-th source"),

	## Independent inputs to E cells
	("/Populations/S/SperE      = 1", "Number of independent spike source per E cell"),
	("/Populations/S/n          = @/Populations/S/SperE@*@/Populations/E/n@", "Total number of spike sources in an external input"),
	("/Populations/S/DistMean   = 1", "Mean firing rate input to one E cell"),

	## Independent inputs to I cells
	#("/Populations/S/PyMeanFr   = 500", "Mean firing rate of one pyramidal cell"),	
	#("/Populations/S/NumberInpus= 300", "number of inputs from pyramidal cell to one interneuron"),	
	#("/Populations/S/SperI      = 1", "Number of independent spike source per I cell"),	
	#("/Populations/S/DistMean   = @/Populations/S/PyMeanFr@/@/Populations/S/NumberInpus@*@/Populations/S/SperI@", "Mean firing mean rate input to one I cell"),
	#("/Populations/S/n          = @/Populations/S/SperI@*@/Populations/I/n@", "Total number of spike sources in an external input"),

	("/Populations/S/spikesource= lambda i, tpre: tpre+random.exponential(scale=@/Populations/S/DistMean@)",
		"Function which generate next spike on i-th source"),
	("/Populations/S/model      = \'from SGen import Sg as NeuronS\'", "Model of single spike source fo stimulus"),
	("/Populations/S/modelname = \'NeuronS\'", "Name of neuron class for S- population"),
	("/Populations/S/init        = zip( range(@/Populations/S/n@), random.random(@/Populations/S/n@)*100. )", "Initiation of time spike generator"),
	("/Populations/S/activate   = [ getspikesequence(sid, start, @/Simulation/stop@, @/Populations/S/spikesource@) for sid,start in enumerate(random.random(@/Populations/S/n@)*100.) ]",
		"Sequence of parameter same size as size of population, which will send as parameter for activate function"),
	
	#  Connectivity
	##  Type of connection, source and destination populations and synapse for connection
	("/Connections/EE/Type        = \'E\'", "Type of connection E/I/G = excitatory/inhibitory/gap-junction"),
	("/Connections/EE/source      = \'E\'", "Name of source for this connection"),
	("/Connections/EE/destination = \'E\'", "Name of destination for this connection"),
	("/Connections/EE/synapse     = \'esyn\'", "Name of synapses on destination neuron"),
	##  Probability function for connections
	("/Connections/EE/P           = 0.004*5000./@/ATotalCell@", "Probability of E->E connections"),
	("/Connections/EE/P-func      = lambda pre, post: random.rand() < @/Connections/EE/P@ and post != pre", "Function which defines when connection  between EE cells will be made"),
	##  Distributions for gmax and delay
	("/Connections/EE/gmax-mean  = 0.00002", "Mean peak conductance between E->E neurons"),
	("/Connections/EE/gmax-stdr  = 0.000", "Standard deviation of peak conductance between E->E neurons"),
	("/Connections/EE/gmax-func  = lambda pre, post:@/Function/TrancatedGauss@(@/Connections/EE/gmax-mean@, @/Connections/EE/gmax-stdr@,  0.)", "Function for conductance distribution"),
	("/Connections/EE/delay-mean = 1.3", "Mean peak conductance delay between E->E neurons"),
	("/Connections/EE/delay-stdr = 0.", "Standard deviation of peak conductance delay between E->E neurons"),
	("/Connections/EE/delay-func = lambda pre, post:@/Function/TrancatedGauss@($/Connections/EE/delay-mean$, $/Connections/EE/delay-stdr$, 0.3)", "Function for delay distribution"),

	("/Connections/EI/Type        = \'E\'", "Type of for E->I connection "),
	("/Connections/EI/source      = \'E\'", "Name of source for E->I connection"),
	("/Connections/EI/destination = \'I\'", "Name of destination for E->I connection"),
	("/Connections/EI/synapse     = \'esyn\'", "Name of synapses for E->I connection"),
	("/Connections/EI/P           = 0.016*5000./@/ATotalCell@", "Probability of E->I connections"),
	("/Connections/EI/P-func     = lambda pre, post: random.rand() < @/Connections/EI/P@                ", "Function which defines when connection  between EI cells will be made"),
	("/Connections/EI/gmax-mean  = 0.0000031", "Mean peak conductance betwEIn E->I neurons"),
	("/Connections/EI/gmax-stdr  = 0.0000", "Standard deviation of peak conductance betwEIn E->I neurons"),
	("/Connections/EI/gmax-func  = lambda pre, post:@/Function/TrancatedGauss@($/Connections/EI/gmax-mean$, $/Connections/EI/gmax-stdr$,  0.)", "Function for conductance distribution"),
	("/Connections/EI/delay-mean = 1.3", "Mean peak conductance delay betwEIn E->I neurons"),
	("/Connections/EI/delay-stdr = 0.", "Standard deviation of peak conductance delay betwEIn E->I neurons"),
	("/Connections/EI/delay-func = lambda pre, post:@/Function/TrancatedGauss@($/Connections/EI/delay-mean$, $/Connections/EI/delay-stdr$, 0.3)", "Function for delay distribution"),

	("/Connections/IE/Type        = \'I\'", "Type of for I->E connection "),
	("/Connections/IE/source      = \'I\'", "Name of source for I->E connection"),
	("/Connections/IE/destination = \'E\'", "Name of destination for I->E connection"),
	("/Connections/IE/synapse     = \'isyn\'", "Name of synapses for I->E connection"),
	("/Connections/IE/P          = 0.016*5000./@/ATotalCell@", "Probability of I->E connections"),
	("/Connections/IE/P-func     = lambda pre, post: random.rand() < @/Connections/IE/P@                ", "Function which defines when connection  between IE cells will be made"),
	("/Connections/IE/gmax-mean  = 0.000016", "Mean peak conductance betwIEn I->E neurons"),
	("/Connections/IE/gmax-stdr  = 0.00", "Standard deviation of peak conductance betwIEn I->E neurons"),
	("/Connections/IE/gmax-func  = lambda pre, post:@/Function/TrancatedGauss@($/Connections/IE/gmax-mean$, $/Connections/IE/gmax-stdr$,  0.)", "Function for conductance distribution"),
	("/Connections/IE/delay-mean = 1.3", "Mean peak conductance delay betwIEn I->E neurons"),
	("/Connections/IE/delay-stdr = 0.", "Standard deviation of peak conductance delay betwIEn I->E neurons"),
	("/Connections/IE/delay-func = lambda pre, post:@/Function/TrancatedGauss@($/Connections/IE/delay-mean$, $/Connections/IE/delay-stdr$, 0.3)", "Function for delay distribution"),

	("/Connections/II/Type        = \'I\'", "Type of for I->I connection "),
	("/Connections/II/source      = \'I\'", "Name of source for I->I connection"),
	("/Connections/II/destination = \'I\'", "Name of destination for I->I connection"),
	("/Connections/II/synapse     = \'isyn\'", "Name of synapses for I->I connection"),
	("/Connections/II/P           = 0.064*5000./@/ATotalCell@", "Probability of I->I connections"),
	("/Connections/II/P-func      = lambda pre, post: random.rand() < @/Connections/II/P@ and post != pre", "Function which defines when connection  between II cells will be made"),
	("/Connections/II/gmax-mean   = 0.000004", "Mean peak conductance betwIIn I->I neurons"),
	("/Connections/II/gmax-stdr   = 0.00", "Standard deviation of peak conductance betwIIn I->I neurons"),
	("/Connections/II/gmax-func   = lambda pre, post:@/Function/TrancatedGauss@($/Connections/II/gmax-mean$, $/Connections/II/gmax-stdr$,  0.)", "Function for conductance distribution"),
	("/Connections/II/delay-mean  = 1.3", "Mean peak conductance delay betwIIn I->I neurons"),
	("/Connections/II/delay-stdr  = 0.", "Standard deviation of peak conductance delay betwIIn I->I neurons"),
	("/Connections/II/delay-func  = lambda pre, post:@/Function/TrancatedGauss@($/Connections/II/delay-mean$, $/Connections/II/delay-stdr$, 0.3)", "Function for delay distribution"),
	
	("/Connections/SE/Type        = \'E\'", "Type of for S->E connection "),
	("/Connections/SE/source      = \'S\'", "Name of source for S->E connection"),
	("/Connections/SE/destination = \'E\'", "Name of destination for S->E connection"),
	("/Connections/SE/synapse     = \'esyn\'", "Name of synapses on destination for S->E connection"),
#	("/Connections/SE/pattern     = equalcodegen(@/Populations/S/n@)","Total pattern of source connections"),
#	("/Connections/SE/P-func      = lambda pre, post: @/Connections/SE/pattern@[post][pre] == '1'", "Function which defines when connection  between inputs and the E cells will be made"),
	("/Connections/SE/P-func      = lambda pre, post: pre%@/Populations/E/n@ == post", "Function which defines when connection  between inputs and the E cells will be made"),
	("/Connections/SE/gmax-mean   = 0.000006", "Mean peak conductance betwSEn S->E neurons"),
	("/Connections/SE/gmax-stdr   = 0.000", "Standard deviation of peak conductance betwSEn S->E neurons"),
	("/Connections/SE/gmax-func   = lambda pre, post:@/Function/TrancatedGauss@($/Connections/SE/gmax-mean$, $/Connections/SE/gmax-stdr$,  0.)", "Function for conductance distribution"),
	("/Connections/SE/delay-mean  = 1.3", "Mean peak conductance delay betwSEn S->E neurons"),
	("/Connections/SE/delay-stdr  = 0.", "Standard deviation of peak conductance delay betwSEn S->E neurons"),
	("/Connections/SE/delay-func  = lambda pre, post:@/Function/TrancatedGauss@($/Connections/SE/delay-mean$, $/Connections/SE/delay-stdr$, 0.3)", "Function for delay distribution"),
	
	#("/Connections/SI/Type        = \'E\'", "Type of for S->I connection "),
	#("/Connections/SI/source      = \'S\'", "Name of source for S->I connection"),
	#("/Connections/SI/destination = \'I\'", "Name of destination for S->I connection"),
	#("/Connections/SI/synapse     = \'esyn\'", "Name of synapses on destination for S->I connection"),
	#("/Connections/SI/P-func      = lambda pre, post: pre%@/Populations/I/n@ == post", "Function which defines when connection  between inputs and the E cells will be made"),
	#("/Connections/SI/gmax-mean   = 0.000031", "Mean peak conductance between S->I neurons"),
	#("/Connections/SI/gmax-stdr   = 0.000", "Standard deviation of peak conductance between S->I neurons"),
	#("/Connections/SI/gmax-func   = lambda pre, post:@/Function/TrancatedGauss@($/Connections/SI/gmax-mean$, $/Connections/SI/gmax-stdr$,  0.)", "Function for conductance distribution"),
	#("/Connections/SI/delay-mean  = 1.3", "Mean peak conductance delay between S->I neurons"),
	#("/Connections/SI/delay-stdr  = 0.", "Standard deviation of peak conductance delay between S->E neurons"),
	#("/Connections/SI/delay-func  = lambda pre, post:@/Function/TrancatedGauss@($/Connections/SI/delay-mean$, $/Connections/SI/delay-stdr$, 0.3)", "Function for delay distribution"),

	
	## Balancing and Scaling
	('/Populations/E/E-Scaling = False', "scaling wight of synapses like n-excitatory connections = /Populations/E/E-Scaling or False to disable scaling"),
	('/Populations/E/I-Scaling = False', "scaling wight of synapses like n-inhibitory connections = /Populations/E/I-Scaling or False to disable scaling"),
	('/Populations/I/E-Scaling = False', "scaling wight of synapses like n-excitatory connections = /Populations/I/E-Scaling or False to disable scaling"),
	('/Populations/I/I-Scaling = False', "scaling wight of synapses like n-inhibitory connections = /Populations/I/I-Scaling or False to disable scaling"),
	('/Populations/E/E2I-Balance-Scaler = 1.5',  "Ratio E/I"),
	('/Populations/I/E2I-Balance-Scaler = 1.',   "Ratio E/I"),
	('/Populations/E/E2I-Balance = lambda EGmax,IGmax,E2I: ( $/Populations/E/E2I-Balance-Scaler$*(EGmax*E2I+IGmax)/E2I/2./EGmax, (EGmax*E2I+IGmax)/IGmax/2. )', 
		"lambda function for E/I balance in each neuron get total max conductance of excitation, total max inhibition and ratio of space under a curve for excitation and inhibition. False - to disable"),
	('/Populations/I/E2I-Balance = lambda EGmax,IGmax,E2I: ( $/Populations/I/E2I-Balance-Scaler$*(EGmax*E2I+IGmax)/E2I/2./EGmax, (EGmax*E2I+IGmax)/IGmax/2. )', 
		"lambda function for E/I balance in each neuron get total max conductance of excitation, total max inhibition and ratio of space under a curve for excitation and inhibition. False - to disable"),
	('/Populations/E/E2I-Balance-Conductance = False',  "If True Balance conductance, if False - current"),
	('/Populations/I/E2I-Balance-Conductance = False',  "If True Balance conductance, if False - current"),
	
	
	#  Presimulation analysis
	("/Analysis/Presim/connectome = True", "Print out connectome statistics"),
	
	#  Postsimulation analysis
	("/Analysis/Postsim/E/PeakDetector/binsize = 1.0", "Binsize for E-population firing rate in ms"),
	("/Analysis/Postsim/E/PeakDetector/kernel  = 5.0", "Sigma of Gausian kernel for E-population in binsize"),
	("/Analysis/Postsim/E/PeakDetector/window  = 25" , "1/2 of window size for E-population in binsize"),
	("/Analysis/Postsim/E/PeakDetector/pure    = False", "Set True if additional check that two maximums are separated by minimum must be appalled"), 
	("/Analysis/Postsim/I/PeakDetector/binsize = 1.0", "Binsize for I-population firing rate in ms"),
	("/Analysis/Postsim/I/PeakDetector/kernel  = 5.0", "Sigma of Gausian kernel for I-population in binsize"),
	("/Analysis/Postsim/I/PeakDetector/window  = 25" , "1/2 of window size for I-population in binsize"),
	("/Analysis/Postsim/I/PeakDetector/pure    = False", "Set True if additional check that two maximums are separated by minimum must be appalled"),
	("/Analysis/Postsim/S/PeakDetector/binsize = 1.0", "Binsize for I-population firing rate in ms"),
	("/Analysis/Postsim/S/PeakDetector/kernel  = 5.0", "Sigma of Gausian kernel for I-population in binsize"),
	("/Analysis/Postsim/S/PeakDetector/window  = 25" , "1/2 of window size for I-population in binsize"),
	("/Analysis/Postsim/S/PeakDetector/pure    = False", "Set True if additional check that two maximums are separated by minimum must be appalled"),
	("/Analysis/Postsim/E/R2                   = True", "Calculate R2 for E population"),
	("/Analysis/Postsim/E/Against/I/R2         = True", "Calculate R2 for E population against peaks of I population spikerate"),
	("/Analysis/Postsim/E/Against/S/R2         = True", "Calculate R2 for E population against peaks of I population spikerate"),
	("/Analysis/Postsim/I/R2                   = True", "Calculate R2 for I population"),
	("/Analysis/Postsim/I/Against/E/R2         = True", "Calculate R2 for I population against peaks of E population spikerate "),
	("/Analysis/Postsim/I/Against/S/R2         = True", "Calculate R2 for I population against peaks of E population spikerate "),
	("/Analysis/Postsim/E/CircDistr            = False", "Phase distribution of E-neurons spikes around a spike-rate cycle (bin size in radians or False)"),
	("/Analysis/Postsim/E/Against/I/CircDistr  = False", "Phase distribution of E-neurons spikes around a spike-rate cycle of I population (bin size in radians or False)"),
	("/Analysis/Postsim/I/CircDistr            = False", "Phase distribution of I-neurons spikes around a spike-rate cycle (bin size in radians or False)"),
	("/Analysis/Postsim/I/Against/E/CircDistr  = False", "Phase distribution of I-neurons spikes around a spike-rate cycle of E population (bin size in radians or False)"),
	("/Analysis/Postsim/E/TaS                  = False", "Calcuating  Tiesinga & Sejnowski metrix for E population"),
	("/Analysis/Postsim/I/TaS                  = False", "Calcuating  Tiesinga & Sejnowski metrix for I population"),
	("/Analysis/Postsim/E/MeanFiringRate       = True",  "Calculate mena firing rate for neurons in E population"),
	("/Analysis/Postsim/I/MeanFiringRate       = True",  "Calculate mena firing rate for neurons in I population"),
	("/Analysis/Postsim/S/MeanFiringRate       = True",  "Calculate mena firing rate for neurons in S population"),
	("/Analysis/Postsim/E/AsymmetryDetector    = True", "Detector of Asymmetry in E-population firing rate"),
	("/Analysis/Postsim/I/AsymmetryDetector    = True", "Detector of Asymmetry in I-population firing rate"),
	("/Analysis/Postsim/E/Balance             = True", "Analysis of current and conductance balance in E neurons"),
	("/Analysis/Postsim/E/Balance/cur/exc     = \'ei\'",  "name of recorder for excitatory current     in E population"),
	("/Analysis/Postsim/E/Balance/cur/inh     = \'ii\'",  "name of recorder for inhibitory current     in E population"),
	("/Analysis/Postsim/E/Balance/con/exc     = \'eg\'",  "name of recorder for excitatory conductance in E population"),
	("/Analysis/Postsim/E/Balance/con/inh     = \'ig\'",  "name of recorder for inhibitory conductance in E population"),
	("/Analysis/Postsim/I/Balance             = True", "Analysis of current and conductance balance in I neurons"),
	("/Analysis/Postsim/I/Balance/cur/exc     = \'ei\'",  "name of recorder for excitatory current     in I population"),
	("/Analysis/Postsim/I/Balance/cur/inh     = \'ii\'",  "name of recorder for inhibitory current     in I population"),
	("/Analysis/Postsim/I/Balance/con/exc     = \'eg\'",  "name of recorder for excitatory conductance in I population"),
	("/Analysis/Postsim/I/Balance/con/inh     = \'ig\'",  "name of recorder for inhibitory conductance in I population"),

	#  Visualization
	("/GUI=True", "Turn on GUI"),
#	('/View/E/volt = \'v\' ', 'View voltage in first plot'),
	#  Debug functions
	("/Debug/print/methods    = True" , "Print out Methods"),
	("/Debug/check/condisfun  = False", "Check all Functions for connections distribution"),
	("/Debug/print/connectome = False", "Print out connectome"),
	("/Debug/check/balance    = False", "Print blance in each neuron"),
	("/Debug/check/sourcedist = False", "Check source distribution"),
	("/Debug/print/netcons = False", "Check source distribution"),

	#  Logger option (don't work)
#	('/CONFIG/LOG/File = Flase', 'Set up logger file'),
#	('/CONFIG/LOG/Level = DEBUG', 'Set up logger level (DEBUG, INFO, CRITICAL')

	#  DBRECORD
	('/CONFIG/simdb = \"pirping-network.simdb\"', "SimDB file name"),
	('/CONFIG/config = \"pirping-network.conf\"', "Generate flat config file")

	]

methods = simmethods(presets=preset, default = defaults, argvs = sys.argv[1:], target="methods", localcontext=globals() )
if reduce(lambda x,y:x or y=="-h" or y=="--help",sys.argv[1:],False):
	print __doc__
	print 
	print "USAGE: nrngui -nogui -python pirping-network.py [parameters]"
	print methods.printhelp()
	exit(0)
	

if "/FROMDBTOCONFIG" in methods.namespace:
	with open(methods.namespace["/FROMDBTOCONFIG"],"w") as fd:
		for name in methods.namespace:
			fd.write("{}={}\n".format(name,methods.namespace[name]) )
	exit(0)

hpc = h.ParallelContext()
print "==================================================="
print "===              Generate METHODS               ==="
methods.generate()
print "===================================================\n"

if methods is None: exit(0)

if methods.check("/Debug/print/methods"):
	print "==================================================="
	print "===                   METHODS                   ==="
	print methods.printmethods(space=" > ")
	print "===================================================\n"
#DB>>
#for dep in methodsgen.hashspace:
	#print dep, ":", methodsgen.hashspace[dep]
#exit(0)

#print methods.gendbrecord()
#exit(0)
#<<DB
if methods.check("/GUI"):
	import matplotlib
	matplotlib.rcParams["savefig.directory"] = ""
	if methods.check("/GUI/Save"): matplotlib.use('Agg')
	from matplotlib.pyplot import *
	import matplotlib.mlab as mlab
	import matplotlib.image as img

if methods.check("/Debug/check/condisfun") and methods.check("/GUI"):
	print "==================================================="
	print "===         CHECK DISTRIBUTION GENERATORS       ==="
	print "===================================================\n"
	ncondistplot = len(dict(methods["/Connections"]))
	for icon,conname in enumerate(methods["/Connections"].dict()):
		ncondistpref, ncondistgmax, ncondistdelay ="/Connections/", "/gmax-func", "/delay-func"
		x = [ methods[ncondistpref+conname+ncondistgmax ](0.,0.) for i in xrange(1000) ]
		y = [ methods[ncondistpref+conname+ncondistdelay](0.,0.) for i in xrange(1000) ]
		subplot(ncondistplot,2,icon*2+1)
		title("Gmax {}".format(conname))
		hist(x,100)
		subplot(ncondistplot,2,icon*2+2)
		title("Delay {}".format(conname))
		hist(y,100)
	show()
	
if methods.check("/Debug/check/sourcedist") and methods.check("/GUI") and methods.check("/Populations/S/spikesource"):
	x = [ methods["/Populations/S/spikesource"](i,0) for i in xrange(1000) ]
	print " > ISI Distribution      : ", np.mean(x),np.std(x)
	hist(x,100)
	show()
	exit(0)

print "==================================================="
print "===                  CONNECTOME                 ==="
if methods.check("/Connectome"):
	print "===             OLD VERSION                 ==="
	print " v /Connectome HAS BEEN READ FROM METHODS"
	if methods.check("/Connectome/EE"):
		print " |-> E->E                           :       FOUND"
		methods["/Connections/EE/Connectome"] = list(methods["/Connectome/EE"])
	if methods.check("/Connectome/EI"):
		print " |-> E->I                           :       FOUND"
		methods["/Connections/EI/Connectome"] = list(methods["/Connectome/EI"])
	if methods.check("/Connectome/IE"):
		print " |-> I->E                           :       FOUND"
		methods["/Connections/IE/Connectome"] = list(methods["/Connectome/IE"])
	if methods.check("/Connectome/II"):
		print " |-> I->I                           :       FOUND"
		methods["/Connections/II/Connectome"] = list(methods["/Connectome/II"])
	if methods.check("/Connectome/SE"):
		print " `-> S->E                           :       FOUND"
		methods["/Connections/SE/Connectome"] = list(methods["/Connectome/SE"])
	methods["/Connectome"] = None

for connect in methods['/Connections'][None]:
	if connect == "<HASH>"                       : continue
	if not methods.check('/Connections/'+connect): continue
	conname = '/Connections/'+connect+'/'
	hashsum = methods['#'+conname+'gmax-func'] + methods['#'+conname+'delay-func']  +\
			  methods['#'+conname+'source']    + methods['#'+conname+'destination'] +\
			  methods['#'+conname+'P-func']
	if methods.check('/Connections/<HASH>/'+connect):
		generate = methods['/Connections/<HASH>/'+connect] == hashsum
	else:
		generate =  False
	
	print ' > {} Connection {} -> {}             : '.format(connect,methods[conname+'source'],methods[conname+'destination']),
	if not generate:
		##DB>>
		#print
		#print [ n for n in methods['/Connections'][None] ]
		#print methods['/Connections/<HASH>']
		#print methods['/Connections/<HASH>/'+connect]
		#print hashsum
		#exit(0)
		##<<DB
		print ' hash sun is different > generate :',
	if methods.check(conname+"Connectome") and generate:
		print ' Connectopm is found'
		continue
	methods[conname+"Connectome"] = [ (pre,post,methods[conname+"gmax-func"](pre,post),methods[conname+"delay-func"](pre,post) ) 
			for pre  in xrange(methods["/Populations/"+methods[conname+'source']     +"/n"]) 
			for post in xrange(methods["/Populations/"+methods[conname+'destination']+"/n"]) 
			if methods[conname+"P-func"](pre,post) ]
	methods['/Connections/<HASH>/'+connect] = hashsum
	print "Done"
print "===================================================\n"


print "==================================================="
print "===           CREATING the Populations          ==="
populations = {}
for pop in methods['/Populations'][None]:
	popname = '/Populations/'+pop+"/"
	if methods.check(popname+'model'):
		cmd = methods[popname+"model"]
		print " > %s: % 40s :"%(pop,cmd),
		try:
			exec cmd
		except BaseException as e:
			raise ValueError("\nCannot import neuron model for {} population: {}.".format(pop,e))
		print "OK"
	else:
		try:
			exec methods[popname+"modelname"]
		except BaseException as e:
			raise ValueError("Cannot find Neuron class {} population {}: {}.".format(methods[popname+"model"], pop, e))

	print " > %s: % 40s :"%(pop,"create the population"),	
	if methods.check(popname+"init"):
		if type(methods[popname+"init"]) is float or type(methods[popname+"init"]) is int:
			methods[popname+"init"] = float(methods[popname+"init"])*ones(methods[popname+"n"])
		elif type(methods[popname+"init"]) is list or type(methods[popname+"init"]) is tuple:
			if len(list(methods[popname+"init"])) != methods[popname+"n"]:
				raise ValueError("Size of {}/init is not equal to {}/n".format(popname,popname))
		elif isinstance(methods[popname+"init"],ndarray):
			if methods[popname+"init"].shape[0] != methods[popname+"n"]:
				raise ValueError("Shape[0] of {}/init is not equal to {}/n".format(popname,popname))
		try:
			exec 'populations[\'{}\'] = [ {}(init=init) for init in methods[popname+"init"] ]'.format(pop,methods[popname+"modelname"])
		except BaseException as e:
			raise ValueError("Cannot create {} population: {}\n ERROR: {}.".format(pop, 
				'populations[\'{}\'] = [ {}(init=init) for init in methods[popname+"init"] ]'.format(pop,methods[popname+"modelname"]), 
				e))
	else:
		try:
			exec 'populations[\'{}\'] = [ {}() for init in xrange(methods[popname+"n"]) ]'.format(pop,methods[popname+"modelname"])
		except BaseException as e:
			raise ValueError("Cannot create {} population: {}.".format(pop, e))
	if methods.check(popname+"params"):
		for param in methods[popname+"params"][None]:
			for n in populations[pop]:
				if type(methods[popname+"params/"+param]) is str:
					try:
						exec "n.{}=".format(param) + methods[popname+"params/"+param]
					except BaseException as e:
						raise RuntimeError("Cannot set parameter {} for neuron in population {}: {}.".format(param,pop, e))
				else:
					try:
						exec "n.{}={}".format(param,methods[popname+"params/"+param])
					except BaseException as e:
						raise RuntimeError("Cannot set parameter {} for neuron in population {}: {}.".format(param,pop, e))
		print " - parameters set -",
	print "OK"
	#>> Debug
	if methods.check(popname+"DB>>/vinit"):
		for n in populations[pop]:
			print n.soma(0.5).v,
		print
	#<< Debug
	if methods.check(popname+"record"):
		print " > %s: % 40s :"%(pop,"set recorders"),
		for n in populations[pop]:
			n.setrecorder(dict(methods[popname+"record"]))
		print "OK"	
	# E Scaling
	if methods.check(popname+"E-Scaling"):
		if not type(methods[popname+"E-Scaling"]) is int and not type(methods[popname+"E-Scaling"]) is float:
			raise ValueError("{}E-Scaling much be integer or float number. {} given".format(popname,type(methods[popname+'E-Scaling'])))
		print " > %s: % 40s :"%(pop,"scaling excitatory connections"),
		for ni,n in enumerate(populations[pop]):
			conin = 0
			conls = []
			for con in methods["/Connections"][None]:
				if con == "<HASH>" : continue
				if methods["/Connections/"+con+"/destination"] != pop or methods["/Connections/"+con+"/Type"] != 'E': continue
				conin += len(where(array(methods["/Connections/"+con+"/Connectome"])[:,1] == ni)[0])
				conls.append(con)
			multy = float(methods[popname+'E-Scaling'])#/float(conin)
			for con in conls:
				methods["/Connections/"+con+"/Connectome"] = [
					x if x[1] != ni else (x[0],x[1],x[2]*multy,x[3]) for x in methods["/Connections/"+con+"/Connectome"]
				]
		print "OK"
		# Set to False to prevent rescaling when read from simdb
		#methods[popname+'E-Scaling'] = False
		# Remove set to false for scaling to document scaler.
		# If reads from simdb should be set at false to revent double scaling

	# I Scaling
	if methods.check(popname+"I-Scaling"):
		if not type(methods[popname+"I-Scaling"]) is int and not type(methods[popname+"I-Scaling"]) is float:
			raise ValueError("{}I-Scaling much be integer or float number. {} given".format(popname,type(methods[popname+'I-Scaling'])))
		print " > %s: % 40s :"%(pop,"scaling inhibitory connections"),
		for ni,n in enumerate(populations[pop]):
			conin = 0
			conls = []
			for con in methods["/Connections"][None]:
				if con == "<HASH>" : continue
				if methods["/Connections/"+con+"/destination"] != pop or methods["/Connections/"+con+"/Type"] != 'I': continue
				conin += len(where(array(methods["/Connections/"+con+"/Connectome"])[:,1] == ni)[0])
				conls.append(con)
			multy = float(methods[popname+'I-Scaling'])#/float(conin)
			for con in conls:
				methods["/Connections/"+con+"/Connectome"] = [
					x if x[1] != ni else (x[0],x[1],x[2]*multy,x[3]) for x in methods["/Connections/"+con+"/Connectome"]
				]
		print "OK"
		# Set to False to prevent rescaling when read from simdb
		#methods[popname+'I-Scaling'] = False
		# Remove set to false for scaling to document scaler.
		# If reads from simdb should be set at false to revent double scaling

	if methods.check(popname+"E2I-Balance"):
		print " > %s: % 40s :"%(pop,"balancing E/I ratio"),
		if methods.check('/Debug/check/balance'): print "DEBUG"
		fn = methods[popname+'E2I-Balance']
		for ni,n in enumerate(populations[pop]):
			E2I = n.getEtoIspace(conduct=methods.check(popname+'E2I-Balance-Conductance'))
			conEin,conIin = 0.,0.
			conEls,conIls = [],[]
			for con in methods["/Connections"][None]:
				if con == "<HASH>" : continue
				if methods["/Connections/"+con+"/destination"] != pop : continue
				if   methods["/Connections/"+con+"/Type"] == 'E':
					conEin += len(where(array(methods["/Connections/"+con+"/Connectome"])[:,1] == ni)[0])
					conEls.append(con)					
				elif methods["/Connections/"+con+"/Type"] == 'I':
					conIin += len(where(array(methods["/Connections/"+con+"/Connectome"])[:,1] == ni)[0])
					conIls.append(con)					
			if conEin < 0.6 or conIin < 0.6:
				scaleE, scaleI = 1., 1.
			else:
				scaleE, scaleI = fn(conEin,conIin, E2I)
			#DB>>
			if methods.check('/Debug/check/balance'):
				print "DB: %s#%03d Ne=%d Ni=%d, E/I=%g >> Se=%g, Si=%g"%(pop,ni,conEin,conIin,E2I,scaleE, scaleI)
			#<<DB
			for con in conEls:
				methods["/Connections/"+con+"/Connectome"] = [
					x if x[1] != ni else (x[0],x[1],x[2]*scaleE,x[3]) for x in methods["/Connections/"+con+"/Connectome"]
				]
			for con in conIls:
				methods["/Connections/"+con+"/Connectome"] = [
					x if x[1] != ni else (x[0],x[1],x[2]*scaleI,x[3]) for x in methods["/Connections/"+con+"/Connectome"]
				]

		if not methods.check('/Debug/check/balance'):
			print "OK"
		methods[popname+'E2I-Balance'] = False
	print
print "===================================================\n"


if methods.check("/Debug/print/connectome"):
	print "==================================================="
	print "===                   CONNECTOME                ==="
	for conname in methods["/Connections"][None]:
		constr = "/Connections/"+conname
		print "{}->{}: Pre \tPost \tGmax   \tDelay".format(methods[constr+'/source'],methods[constr+'/destination'])
		for con in methods[constr+"/Connectome"]:
			print "{}->{}: % 3d\t% 3d\t%e\t%e".format(methods[constr+'/source'],methods[constr+'/destination'])%con
		print
	print "===================================================\n"


if methods.check("/Analysis/Presim/connectome"):
	print "==================================================="
	print "====           CONNECTOME  STATISTICS          ===="
	for con in methods["/Connections"][None]:
		if con == "<HASH>"                     : continue
		if not methods.check('/Connections/'+con): continue
		conname = "/Connections/"+con+'/'
		Xcon = array(methods[conname+"Connectome"])
		if Xcon.shape[0] < 2: continue
		pre = array([ len(where(Xcon.astype(int)[:,0]==pre)[0]) for pre in xrange(methods["/Populations/"+methods[conname+"source"]+"/n"]) ])
		pst = array([ len(where(Xcon.astype(int)[:,1]==pst)[0]) for pst in xrange(methods["/Populations/"+methods[conname+"destination"]+"/n"]) ])
		methods["/Analysis/Presim/connectome/"+con+"/preNcon/mean"]  = mean(pre)
		methods["/Analysis/Presim/connectome/"+con+"/preNcon/stdr"]  = std(pre)
		methods["/Analysis/Presim/connectome/"+con+"/postNcon/mean"] = mean(pre)
		methods["/Analysis/Presim/connectome/"+con+"/postNcon/stdr"] = std(pre)
		methods["/Analysis/Presim/connectome/"+con+"/gmax/mean"]     = mean(Xcon[:,2])
		methods["/Analysis/Presim/connectome/"+con+"/gmax/stdr"]     = std(Xcon[:,2])
		methods["/Analysis/Presim/connectome/"+con+"/delay/mean"]    = mean(Xcon[:,3])
		methods["/Analysis/Presim/connectome/"+con+"/delay/stdr"]    = std(Xcon[:,3])
		print " {}:{}->{} number connections presyn.    : {}+-{}".format(con,methods[conname+"source"],methods[conname+"destination"], mean(pre),std(pre))
		print " {}:{}->{} number connections postsyn.   : {}+-{}".format(con,methods[conname+"source"],methods[conname+"destination"], mean(pst),std(pst))
		print " {}:{}->{} max synaptic condactance Gsyn : {}+-{}".format(con,methods[conname+"source"],methods[conname+"destination"], mean(Xcon[:,2]),std(Xcon[:,2]))
		print " {}:{}->{} synaptic delay                : {}+-{}".format(con,methods[conname+"source"],methods[conname+"destination"], mean(Xcon[:,3]),std(Xcon[:,3]))
		print
	print "===================================================\n"

print "==================================================="
print "===             CREATING the Network            ==="
netcons={}
for conname in methods["/Connections"][None]:
	if conname == "<HASH>"                       : continue
	if not methods.check('/Connections/'+conname): continue
	source      = methods['/Connections/'+conname+"/source"]
	destination = methods['/Connections/'+conname+"/destination"]
	synapse     = methods['/Connections/'+conname+"/synapse"]
	print " > Connections % 31s :"%("from {} to {}".format(source,destination)),
	netcons[conname] = []
	for pre,post,gsyn,delay in methods['/Connections/'+conname+"/Connectome"]:
		try:
			exec "netcons[\'{}\'].append(h.NetCon(populations[\'{}\'][{}].output, populations[\'{}\'][{}].{}, 0., delay, gsyn, sec=populations[\'{}\'][pre].soma))".format(
				conname, source, pre, destination, post, synapse, source, pre)
		except BaseException as e:
			raise ValueError("Cannot create {} connection: {}.".format(conname, e))		
	
	print "OK"
print "===================================================\n"

if methods.check('/Debug/print/netcons'):
	for conname in methods["/Connections"][None]:
		for con in netcons[conname]:
			print "{}: WEIGHT={} DELAT={}".format(conname,con.weight[0],con.delay)

print "==================================================="
print "===            ACTIVATING POPULATIONS           ==="
for popname in methods['/Populations'][None]:
	popactivate = '/Populations/'+popname+"/activate"
	if not methods.check('/Populations/'+popname+"/activate") : continue
	print " > Population % 26s is "%(" {} ".format(popname)),
	if not ( type(methods[popactivate]) is tuple or type(methods[popactivate]) is list ):
		raise ValueError("Activate should be a tuple or a list.  {} given.".format(type(methods[popactivate])))
	if len(methods[popactivate]) != methods['/Populations/'+popname+"/n"]:
		raise ValueError("Size of the activate sequence should a size of population.  {} vs n={} given.".format(
			len(methods[popactivate]),methods['/Populations/'+popname+"/n"]))
	for n,seq in zip(populations[popname],methods[popactivate]):
		n.activate(seq)
	print "Active"
#print "===                     DONE                    ==="
print "===================================================\n"

	
print "==================================================="
print "===  CHECK POSTSIM ANALYSIS BEFORE SIMULATION   ==="
for popname in methods['/Populations'].dict():
	aps="/Analysis/Postsim/"
	arainst = reduce(lambda x,y: x or methods.check(aps+y+"/Against/"+popname), methods['/Populations'].dict(), False)
	if not (methods.check(aps+popname) or arainst or methods.check(aps+popname+"/Balance") ): continue
	print " > %s: % 40s :"%(popname,"peak petector is "),
	for check in ['/R2','/CircDistr']:
		if methods.check(aps+popname+check) and not methods.check(aps+popname+'/PeakDetector'):
			print "\nERROR: PeakDetector must be set for {} population for {} analysis\n".format(popname,aps+popname+check)
			exit(1)
		for xpopname in methods['/Populations'].dict():
			if xpopname == popname : continue
			if methods.check(aps+xpopname+"/Against/"+popname+check) and not methods.check(aps+popname+'/PeakDetector'):
				print "\nERROR: PeakDetector must be set for {} population for {} analysis against {} population\n".format(popname,aps++"/Against/"+popname+check, popname)
				exit(1)
	if methods.check(aps+popname+'/PeakDetector'):
		if not methods.check(aps+popname+'/PeakDetector/binsize'):
			print "\nERROR: binsize must be set to sample {}-population firing rate in PeakDetector\n".format(popname)
			exit(1)
		elif not methods.check(aps+popname+'/PeakDetector/kernel'):
			print "\nERROR: kernel size must be set to sample {}-population firing rate in PeakDetector\n".format(popname)
			exit(1)
		elif not methods.check(aps+popname+'/PeakDetector/window'):
			print "\nERROR: window size must be set to sample {}-population firing rate in PeakDetector\n".format(popname)
			exit(1)
	print "OK"
	
	if not methods.check(aps+popname+'/Balance'): continue
	print " > %s: % 40s :"%(popname," current/conductance balance is "),
	for check in ['/cur','/con']:
		if methods.check(aps+popname+'/Balance'+check):
			if not( methods.check(aps+popname+'/Balance'+check+"/exc") and methods.check(aps+popname+'/Balance'+check+"/inh") ):
				print "\nERROR: Names of recorded variables for inhibition /inh and excitation /exc must be set for {}\n".format(aps+popname+'/Balance'+check)
				exit(1)
			if not methods.check("/Populations/"+popname+"/record/"+methods[aps+popname+'/Balance'+check+"/inh"]):
				print "\nERROR: Cannot find inhibitory variables {} in record section of  {} population\n".format(methods[aps+popname+'/Balance'+check+"/inh"],popname)
				exit(1)
			if not methods.check("/Populations/"+popname+"/record/"+methods[aps+popname+'/Balance'+check+"/exc"]):
				print "\nERROR: Cannot find excitatory variables {} in record section of  {} population\n".format(methods[aps+popname+'/Balance'+check+"/inh"],popname)
				exit(1)
			
	print "OK"
	print
print "===================================================\n"

print "==================================================="
print "===                  Taking off                 ==="
print " > Set time recorder                 : ",
t = h.Vector()
t.record(h._ref_t)
print "Done"
if not '/Parallel/mpi' in methods and '/Parallel/cores' in methods:
	print " > Set Parallel Context threading    : ",
	if methods.check('/Parallel/cores/autodetect'):
		if not os.path.exists("/etc/beowulf") and os.path.exists("/sysini/bin/busybox"):
			#I'm not on head node. I can use all cores (^_^)
			methods['/Parallel/cores/number'] = methods['/Parallel/cores/neadnode']
		elif os.path.exists("/etc/beowulf"):
			#I'm on head node. I grub only half (0_0)
			methods['/Parallel/cores/number'] = methods['/Parallel/cores/compute']
		else:
			#I'm on Desktop (-.-)
			methods['/Parallel/cores/number'] = methods['/Parallel/cores/desktop']
	hpc.nthread(methods['/Parallel/cores/number'])
	print methods['/Parallel/cores/number'],"cores"
else:
	print " >  Cannot set parallel multithreading"
	
if methods.check("/Simulation/cvode"):
	cvode = h.CVode()
	cvode.active(1)
	print " > CVODE                             :  ON"

print " > Set integration time step         : ",
if "/Simulation/step" in methods:
	h.dt = methods["/Simulation/step"]
else:
	h.dt = 0.025
print h.dt,"(ms)"

if methods.check('/Simulation/Celsius' ):
	print " > Set Temperature                   : ",
	h.celsius = methods['/Simulation/Celsius']
	print h.celsius, "Celsius"
	
print " > Engine ignition                   : ",
h.finitialize()
h.fcurrent()
h.frecord_init()
print "OK"
print "===================================================\n"

if not methods.check('/Simulation/norun'):
	print "==================================================="
	print "===                     RUN                     ==="
	tstop = methods['/Simulation/stop']
	while h.t < tstop :h.fadvance()
	print "===================================================\n"
else:
	print "<                      NO RUN                     >"
	if methods.check('/CONFIG/config'):
		confile = str(methods['/CONFIG/config'])
		methods['/CONFIG/config'] = False
		methods['/Simulation/norun'] = False
		#methods.genconfile(confile, unresolved=True)
		methods.genconfile(confile)
	exit(0)

print "==================================================="
print "===           Collect population spikes         ==="
spikes={}
for pop in methods["/Populations"].dict():
	print " > {} population                               : ".format(pop),
	spikes[pop]=[]
	for ni, n in enumerate(populations[pop]):
		spikes[pop] += [ ( st,float(ni) ) for st in n.spks ]
	if len(spikes[pop]) == 0: 
		spikes[pop] =array([ [0.,0] ])
		print "Dummy spike"
	else:
		spikes[pop] = array(sorted(spikes[pop]))
		print "OK"	
print "===================================================\n"


if methods.check("/Analysis"):
	print "==================================================="
	print "===                                             ==="
	print "===                   ANALYSIS                  ==="
	print "===                                             ==="
	print "=== - - - - - - - - - - - - - - - - - - - - - - ==="
	aps = "/Analysis/Postsim/"
	print "===                                             ==="
	firingrate,smoothrate={},{}
	print "===                 Peak Detector               ==="
	for pop in methods["/Analysis/Postsim"].dict():
		if not methods.check(aps+pop+"/PeakDetector"): continue
		print " > %s: % 40s :"%(pop,"collecting firing rate"),
		binsize = methods[aps+pop+'/PeakDetector/binsize']
		frate = zeros( int(ceil(float(methods['/Simulation/stop'])/binsize))+1 )
		for i in spikes[pop][:,0]: frate[int(floor(i/binsize))] += 1
		firingrate[pop] = array(frate)
		print "OK"
		if methods.check(aps+pop+'/Against'):
			for agns in methods[aps+pop+'/Against'].dict():
				print " > %s: % 40s :"%(pop,"{} firing rate against {}".format(agns,pop)),
				if methods[aps+agns+'/PeakDetector/binsize'] == binsize:
					if agns in firingrate:
						firingrate[pop+'/'+agns] = firingrate[agns]
						print "OK"
						continue
					else:
						frate = zeros( int(ceil(float(methods['/Simulation/stop'])/binsize))+1 )
						for i in spikes[agns][:,0]: frate[int(floor(i/binsize))] += 1
						firingrate[pop+'/'+agns] = firingrate[agns] = array(frate)
						print "OK"
						continue
				else:
						frate = zeros( int(ceil(float(methods['/Simulation/stop'])/binsize))+1 )
						for i in spikes[agns][:,0]: frate[int(floor(i/binsize))] += 1
						firingrate[pop+'/'+agns] = array(frate)
						print "OK"
				
		print " > %s: % 40s :"%(pop,"applying kernel"),
		kernel = arange(-methods[aps+pop+'/PeakDetector/window'],methods[aps+pop+'/PeakDetector/window'],1.)
		kernel = exp(kernel**2/methods[aps+pop+'/PeakDetector/kernel']**2*(-1.))
		smooth = convolve(firingrate[pop],kernel)
		smooth = smooth[kernel.size/2:1-kernel.size/2]
		smoothrate[pop] = array( smooth )
		print "OK"
		print " > %s: % 40s :"%(pop,"find maximums and minimums"),
		marks = []
		for idx in (diff(sign(diff(smooth))) < 0).nonzero()[0] + 1: marks.append([idx, 1]) # maximums
		for idx in (diff(sign(diff(smooth))) > 0).nonzero()[0] + 1: marks.append([idx,-1]) # minimums
		marks = array(sorted(marks))
		print "OK"
		print " > %s: % 40s :"%(pop,"sorting peaks"),
		peaks  = []
		pureflag = methods.check(aps+pop+'/PeakDetector/pure')
		for mx in where( marks[:,1] > 0 )[0]:
			if mx <= 2 or mx >= (marks.size/2 -2):continue
			if pureflag and (marks[mx-1][1] > 0 or marks[mx+1][1] > 0 or marks[mx][1] < 0) :continue
			peaks.append(marks[mx][0])
		methods[aps+pop+'/PeakDetector/Peaks'] = list(peaks)
		print "OK"
		print
	print "===                                             ==="
	print "=== - - - - - - - - - - - - - - - - - - - - - - ==="
	print "===                                             ==="
	print "===Circular Statistic and Circular Distributions==="
	for pop in methods["/Analysis/Postsim"].dict():
		netpercnt = 0.
		R2flag = methods.check(aps+pop+'/R2')
		if R2flag:
			X,Y,Rcnt,SPC,netpermean =0.,0.,0.,0.,0.
			frate = firingrate[pop]
		CDflag = methods.check(aps+pop+'/CircDistr')
		if CDflag:
			phydist  = []
			frate = methods[aps+pop+'PeakDetector/FiringRate']
		AGNflag, agnsan = False, {}
		if methods.check(aps+pop+'/Against'):
			for agns in methods[aps+pop+'/Against'].dict():
				agnsan[agns]={}
				agnsan[agns]['R2flag'] = methods.check(aps+pop+'/Against/'+agns+"/R2")
				agnsan[agns]['CDflag'] = methods.check(aps+pop+'/Against/'+agns+"/CircDistr")
				if agnsan[agns]['R2flag'] :
					AGNflag = True
					agnsan[agns]['X'],agnsan[agns]['Y'],agnsan[agns]['Rcnt'],agnsan[agns]['SPC'] = 0.,0.,0.,0.
					agnsan[agns]['frate'] = firingrate[pop+'/'+agns]
				if agnsan[agns]['CDflag'] :
					AGNflag = True
					agnsan[agns]['phydist'] = []
					agnsan[agns]['frate'] = firingrate[pop+'/'+agns]
		if not( R2flag or CDflag or AGNflag) : continue
		print " > %s: % 40s :"%(pop,"R2 and Circular Dist"),
		for l,r in zip(methods[aps+pop+'/PeakDetector/Peaks'][:-1], methods[aps+pop+'/PeakDetector/Peaks'][1:]):
			Pnet = float(r-l)
			netpermean += Pnet*methods[aps+pop+'/PeakDetector/binsize']
			netpercnt  += 1.
			if R2flag or CDflag:
				SPC += sum(frate[l:r])
				for i,n in enumerate(frate[l:r]):
					if R2flag:
						phyX = cos(pi*2.*float(i)/Pnet)
						phyY = sin(pi*2.*float(i)/Pnet)
						X += n*phyX
						Y += n*phyY
						Rcnt += n
					if CDflag: phydist.append( (pi*2.*float(i)/Pnet,n) )
			for agns in agnsan:
				agnsan[agns]['SPC'] += sum(agnsan[agns]['frate'][l:r])
				for i,n in enumerate(agnsan[agns]['frate'][l:r]):
					if agnsan[agns]['R2flag']:
						phyX = cos(pi*2.*float(i)/Pnet)
						phyY = sin(pi*2.*float(i)/Pnet)
						agnsan[agns]['X']    += n*phyX
						agnsan[agns]['Y']    += n*phyY
						agnsan[agns]['Rcnt'] += n
					if agnsan[agns]['CDflag']: 
						phydistI.append( (pi*2.*float(i)/Pnet,n) )
		print "OK"
		if netpercnt > 1.:
			if Rcnt > 0.:
				if R2flag:
					methods[aps+pop+'/R2/R2']            = (X/Rcnt)**2+(Y/Rcnt)**2
					methods[aps+pop+'/R2/SPC']           = SPC/netpercnt
					methods[aps+pop+'/R2/MeanNetPeriod'] = netpermean / ( netpercnt - 1)
					print " > %s: -> % 37s :"%(pop,"R2")             , methods[aps+pop+'/R2/R2']
					print " > %s: -> % 37s :"%(pop,"SPC")            , methods[aps+pop+'/R2/SPC']
					print " > %s: -> % 37s :"%(pop,"Mean Net Period"), methods[aps+pop+'/R2/MeanNetPeriod']
				else:
					methods[aps+pop+'/R2'] = None
				if CDflag:
					print " > %s: -> % 37s :"%(pop,"Phase Distribution")
					phydist = array(phydist)
					if phydist.shape[0] > 0 and len(phydist.shape) > 1:						
						if sum(phydist[:,1]) > 0: 
							phydist = array(phydist)
							phydist[:,1] /= sum(phydist[:,1])
							if not methods.check(aps+pop+'/CircDistr/binsize'):
								methods[aps+pop+'/CircDistr/binsize'] = methods[aps+pop+'/CircDistr']
							phyhist,phyhistbins = histogram(phydist[:,0], bins=int(ceil(pi/methods[aps+pop+'/CircDistr/binsize'])+3), 
								weights=phydist[:,1],
								range=(-pi/methods[aps+pop+'/CircDistr/binsize'],2.*pi+pi/methods[aps+pop+'/CircDistr/binsize']))
							methods[aps+pop+'/CircDistr/histogram']       = phyhist
							methods[aps+pop+'/CircDistr/bins-boundaries'] = phyhistbins 
							print "OK"
						else:
							methods[aps+pop+'/CircDistr'] = None
							print "None"
					else:
						methods[aps+pop+'/CircDistr'] = None
						print "None"
			else:
				print " > %s: -> % 37s :"%(pop,"Rcnt is ZERO"), "Analysis faulted"
				if R2flag: methods[aps+pop+'/R2']        = None
				if CDflag: methods[aps+pop+'/CircDistr'] = None
				#if R2flag: del methods[aps+pop+'/R2'] #= None
				#if CDflag: del methods[aps+pop+'/CircDistr'] #= None
			for agns in agnsan:
				if agnsan[agns]['R2flag']:
					if agnsan[agns]['Rcnt'] > 0.:
						if agnsan[agns]['R2flag']:
							methods[aps+pop+'/Against/'+agns+'/R2/R2']            = (agnsan[agns]['X']/agnsan[agns]['Rcnt'])**2+(agnsan[agns]['Y']/agnsan[agns]['Rcnt'])**2
							methods[aps+pop+'/Against/'+agns+'/R2/Phase']         = arctan2((agnsan[agns]['Y']/agnsan[agns]['Rcnt']), (agnsan[agns]['X']/agnsan[agns]['Rcnt']) )
							methods[aps+pop+'/Against/'+agns+'/R2/SPC']           = agnsan[agns]['SPC']/netpercnt
							print " > %s against %s: -> % 27s :"%(agns,pop,"R2")             , methods[aps+pop+'/Against/'+agns+'/R2/R2']
							print " > %s against %s: -> % 27s :"%(agns,pop,"Phase")          , methods[aps+pop+'/Against/'+agns+'/R2/Phase']
							print " > %s against %s: -> % 27s :"%(agns,pop,"SPC")            , methods[aps+pop+'/Against/'+agns+'/R2/SPC']
						if agnsan[agns]['CDflag']:
							print " > %s against %s: -> % 27s :"%(agns,pop,"Phase Distribution")
							phydist = array(agnsan[agns]['phydist'])
							
							if not methods.check(aps+pop+'/Against/'+agns+'/CircDistr/binsize'):
								methods[aps+pop+'/Against/'+agns+'/CircDistr/binsize'] = methods[aps+pop+'/Against/'+agns+'/CircDistr']
							if phydist.shape[0] > 0 and len(phydist.shape) > 1:						
								if sum(phydist[:,1]) > 0: 
									phydist = array(phydist)
									phydist[:,1] /= sum(phydist[:,1])
									phyhist,phyhistbins = histogram(phydist[:,0], bins=int(ceil(pi/methods[aps+pop+'/Against/'+agns+'/CircDistr/binsize'])+3), 
										weights=phydist[:,1],
										range=(-pi/methods[aps+pop+'/Against/'+agns+'/CircDistr/binsize'],2.*pi+pi/methods[aps+pop+'/Against/'+agns+'/CircDistr/binsize']))
									methods[aps+pop+'/Against/'+agns+'/CircDistr/histogram']       = phyhist
									methods[aps+pop+'/Against/'+agns+'/CircDistr/bins-boundaries'] = phyhistbins 
									print "OK"
								else:
									methods[aps+pop+'/Against/'+agns+'/CircDistr'] = None
									print "None"
							else:
								methods[aps+pop+'/Against/'+agns+'/CircDistr'] = None
								print "None"
					else:
						print " > %s against %s: -> % 27s :"%(agns,pop,"Rcnt is ZERO"), "Analysis faulted"
						if R2flag: del methods[aps+pop+'/Against/'+agns+'/R2']        #= None
						if CDflag: del methods[aps+pop+'/Against/'+agns+'/CircDistr'] #= None
						 

		else:
			print " > %s: % 40s :"%(popname,"Count for period is ZERO"), "Analysis faulted"
		print
	print "===                                             ==="
	print "===---------------------------------------------==="
	print "===                                             ==="
	print "===             Neuron Firing Rate              ==="
	for pop in methods["/Analysis/Postsim"].dict():
		if not methods.check(aps+pop+"/MeanFiringRate"): continue
		print " > %s: % 40s :"%(pop,"neurons mean rate"),
		mrate = float(spikes[pop].shape[0])*1000./float(methods['/Populations/'+pop+'/n']*methods['/Simulation/stop'])
		print mrate, "spike/sec/neuron"
		methods[aps+pop+"/MeanFiringRate"] = mrate
	#DB>>
	#exit(0)
	#<<DB

	if reduce(lambda x,y: methods.check('/Analysis/Postsim/'+y+"/Balance") or x,  methods['/Analysis/Postsim'].dict(), False):
	#if methods.check('/Analysis/Postsim/Balance'):
		print "===                                             ==="
		print "===---------------------------------------------==="
		print "===                                             ==="
		print "===                E/I Balance                  ==="
		for pop in  methods['/Analysis/Postsim'].dict():
			if methods.check('/Analysis/Postsim/'+pop+"/Balance/cur"):
				if methods.check('/Analysis/Postsim/'+pop+"/Balance/cur/exc") and \
				   methods.check('/Analysis/Postsim/'+pop+"/Balance/cur/inh"):
					ii, ie = array([]),array([])
					mi, me = methods['/Analysis/Postsim/'+pop+"/Balance/cur/inh"], methods['/Analysis/Postsim/'+pop+"/Balance/cur/exc"]
					for n in populations[pop]:
						ii = append(ii, array(n.rec[mi]))
						ie = append(ie, array(n.rec[me]))
					methods['/Analysis/Postsim/'+pop+"/Balance/cur/mean-exc"] = mean(ie)
					methods['/Analysis/Postsim/'+pop+"/Balance/cur/mean-inh"] = mean(ii)
					methods['/Analysis/Postsim/'+pop+"/Balance/cur/total"] = mean(ii+ie)
					print " > % 24s mean e-current     : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/cur/mean-exc"]
					print " > % 24s mean i-current     : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/cur/mean-inh"]
					print " > % 24s total  current     : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/cur/total"]
					if methods['/Analysis/Postsim/'+pop+"/Balance/cur/mean-inh"] != 0.0:
						methods['/Analysis/Postsim/'+pop+"/Balance/cur/E2I"] = methods['/Analysis/Postsim/'+pop+"/Balance/cur/mean-exc"]/methods['/Analysis/Postsim/'+pop+"/Balance/cur/mean-inh"]
						print " > % 24s e/i-current        : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/cur/E2I"]

				if methods.check('/Analysis/Postsim/'+pop+"/Balance/con/exc") and \
				   methods.check('/Analysis/Postsim/'+pop+"/Balance/con/inh"):
					ii, ie = array([]),array([])
					mi, me = methods['/Analysis/Postsim/'+pop+"/Balance/con/inh"], methods['/Analysis/Postsim/'+pop+"/Balance/con/exc"]
					for n in populations[pop]:
						ii = append(ii, array(n.rec[mi]))
						ie = append(ie, array(n.rec[me]))
					methods['/Analysis/Postsim/'+pop+"/Balance/con/mean-exc"] = mean(ie)
					methods['/Analysis/Postsim/'+pop+"/Balance/con/mean-inh"] = mean(ii)
					methods['/Analysis/Postsim/'+pop+"/Balance/con/total"] = mean(ie - ii)
					print " > % 24s mean e-conductance : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/con/mean-exc"]
					print " > % 24s mean i-conductance : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/con/mean-inh"]
					print " > % 24s total  conductance : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/con/total"]
					if methods['/Analysis/Postsim/'+pop+"/Balance/con/mean-inh"] != 0.0:
						methods['/Analysis/Postsim/'+pop+"/Balance/con/E2I"] = methods['/Analysis/Postsim/'+pop+"/Balance/con/mean-exc"]/methods['/Analysis/Postsim/'+pop+"/Balance/con/mean-inh"]
						print " > % 24s e/i-conductance    : "%pop, methods['/Analysis/Postsim/'+pop+"/Balance/con/E2I"]

		#Eib,Egb=[],[]
		
			#Eib.append( ( mean( array(n.rec["ei"]) ),mean( array(n.rec["ii"]) ) ) )
			#Egb.append( ( mean( array(n.rec["eg"]) ),mean( array(n.rec["ig"]) ) ) )
		#Iib,Igb=[],[]
		#for n in populations['I']:
			#Iib.append( ( mean( array(n.rec["ei"]) ),mean( array(n.rec["ii"]) ) ) )
			#Igb.append( ( mean( array(n.rec["eg"]) ),mean( array(n.rec["ig"]) ) ) )
		
		#Eib,Egb = array(Eib),array(Egb)
		#Iib,Igb = array(Iib),array(Igb)
		#methods['/Analysis/Postsim/Balance'] = methodtree()
		#methods['/Analysis/Postsim/Balance/Populations/E/E2I-curent' ] = mean(Eib[:,0]/Eib[:,1]),
		#methods['/Analysis/Postsim/Balance/Populations/E/E2I-conduct'] = mean(Egb[:,0]/Egb[:,1])
		#methods['/Analysis/Postsim/Balance/Populations/I/E2I-curent' ] = mean(Iib[:,0]/Iib[:,1]),
		#methods['/Analysis/Postsim/Balance/Populations/I/E2I-conduct'] = mean(Igb[:,0]/Igb[:,1])
		#print " > E-population current         : ",mean(Eib[:,0]/Eib[:,1])
		#print " > E-population conductance     : ",mean(Egb[:,0]/Egb[:,1])
		#print " > I-population current         : ",mean(Iib[:,0]/Iib[:,1])
		#print " > I-population conductance     : ",mean(Igb[:,0]/Igb[:,1])
	N_Spectrum = reduce(lambda x,y: methods.check('/Analysis/Postsim/'+y+"/N_Spectrum") or x,  methods['/Analysis/Postsim'].dict(), False)
	P_Spectrum = reduce(lambda x,y: methods.check('/Analysis/Postsim/'+y+"/P_Spectrum") or x,  methods['/Analysis/Postsim'].dict(), False)
	if P_Spectrum or N_Spectrum:
		print "===                                             ==="
		print "===---------------------------------------------==="
		print "===                                             ==="
		print "===         Network and Neuron Spectrum         ==="
		spnbin = int(np.floor(methods['/Simulation/stop']))+1
		spnbin = 2**int(floor(log2(spnbin)))
		pnum 	= int(200.*methods['/Simulation/stop']/1000.0)
		specX	= np.arange(spnbin, dtype=float) * 1000.0/methods['/Simulation/stop']
		specX	= specX[:pnum]
		for pop in  methods['/Analysis/Postsim'].dict():
			if methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum"):
				specN	= np.zeros(pnum)
			if methods.check('/Analysis/Postsim/'+pop+"/P_Spectrum"):
				spbins  = np.zeros(spnbin)
			for n in populations[pop]:
				if methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum"):
					spn	= np.zeros(spnbin)
					np.add.at(spn,   [ int(np.floor(st)) for st in n.spks if st < spnbin ],1)
					##DB>>
					#if pop == "E":
						#spn = np.sin(2*np.pi*np.arange(spn.shape[0])*10./1000.)
					#elif pop == 'I':
						#spn = np.sin(2*np.pi*np.arange(spn.shape[0])*40./1000.)
					##<<DB
					fft = np.abs( np.fft.fft(spn)*1.0/methods['/Simulation/stop'] )**2
					##DB>>
					#if pop == "E":
						#figure(100)
						#subplot(121)
						#plot(specX,fft[:pnum])
					#elif pop == 'I':
						#figure(100)
						#subplot(122)
						#plot(specX,fft[:pnum])
					##<<DB
					specN  += fft[:pnum]
				if methods.check('/Analysis/Postsim/'+pop+"/P_Spectrum") and methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum"):
					spbins += spn
				elif methods.check('/Analysis/Postsim/'+pop+"/P_Spectrum"):
					np.add.at(spbins,[ int(np.floor(st)) for st in n.spks if st < spnbin ],1)

			if methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum") and methods.check('/Analysis/Postsim/'+pop+"/P_Spectrum") and methods.check('/Analysis/Postsim/'+pop+"/N-P_Spectrum"):
				spbins = spbins.astype(float) / float(methods['/Populations/'+pop+'/n']) #<normolized by size of the population
				fft = np.abs( np.fft.fft(spbins)*1.0/methods['/Simulation/stop'] )**2 
				netsp = np.dstack((specX,fft[:pnum] ))[0]
				methods['/Analysis/Postsim/'+pop+"/P_Spectrum"] = netsp
				print " > % 19s max network spectrum at : "%pop, netsp[argmax(netsp[1:,1])+1,0],"Hz"
				nrnsp = np.dstack((specX, specN / float(methods['/Populations/'+pop+'/n']) - netsp[:,1]))[0]
				methods['/Analysis/Postsim/'+pop+"/N_Spectrum"] = nrnsp
				print " > % 20s max neural spectrum at : "%pop, nrnsp[argmax(nrnsp[1:,1])+1,0],"Hz"
				methods['/Analysis/Postsim/'+pop+"/N_Spectrum_max"] = nrnsp[argmax(nrnsp[1:,1])+1,:]
			elif methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum") and methods.check('/Analysis/Postsim/'+pop+"/P_Spectrum"):
				spbins = spbins.astype(float) / float(methods['/Populations/'+pop+'/n']) 
				fft = np.abs( np.fft.fft(spbins)*1.0/methods['/Simulation/stop'] )**2 
				netsp = np.dstack((specX,fft[:pnum] ))[0]
				methods['/Analysis/Postsim/'+pop+"/P_Spectrum"] = netsp
				print " > % 19s max network spectrum at : "%pop, netsp[argmax(netsp[1:,1])+1,0],"Hz"
				nrnsp = np.dstack((specX, specN / float(methods['/Populations/'+pop+'/n']) ))[0]
				methods['/Analysis/Postsim/'+pop+"/N_Spectrum"] = nrnsp
				print " > % 20s max neural spectrum at : "%pop, nrnsp[argmax(nrnsp[1:,1])+1,0],"Hz"
				methods['/Analysis/Postsim/'+pop+"/N_Spectrum_max"] = nrnsp[argmax(nrnsp[1:,1])+1,:]

			elif methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum"):
				nrnsp = np.dstack((specX, specN / float(methods['/Populations/'+pop+'/n'])))[0]
				methods['/Analysis/Postsim/'+pop+"/N_Spectrum"] = nrnsp
				print " > % 20s max neural spectrum at : "%pop, nrnsp[argmax(nrnsp[1:,1])+1,0],"Hz"
				methods['/Analysis/Postsim/'+pop+"/N_Spectrum_max"] = nrnsp[argmax(nrnsp[1:,1])+1,:]

			elif methods.check('/Analysis/Postsim/'+pop+"/P_Spectrum"):
				spbins = spbins.astype(float) / float(methods['/Populations/'+pop+'/n']) 
				fft = np.abs( np.fft.fft(spbins)*1.0/methods['/Simulation/stop'] )**2 
				netsp = np.dstack((specX,fft[:pnum] ))[0]
				methods['/Analysis/Postsim/'+pop+"/P_Spectrum"] = netsp
				print " > % 19s max network spectrum at : "%pop, netsp[argmax(netsp[1:,1])+1,0],"Hz"
				methods['/Analysis/Postsim/'+pop+"/P_Spectrum_max"] = netsp[argmax(netsp[1:,1])+1,:]
		print "===================================================\n"

if methods.check('/CONFIG/simdb'):
	methods.simdbrecord(zipped=True,\
		message= methods['/CONFIG/simdb-message'] if methods.check('/CONFIG/simdb-message') else None
	)
	#gitdbrec = gitrec(names=['SGen.py','ECellOlufsen.mod','ECellOlufsen.py','Golomb.mod','Golomb.py','HHinh.py','BSKCch.mod','WBinh.py', 'sinIstim.mod','pirping-network.py'])



if methods.check('/GUI'):
	if not (type( methods['/GUI']) is dict or isinstance(methods['/GUI'], methodtree) ): methods['/GUI'] = {'Fcount':1}
	if not methods.check('/GUI/Fcount'): methods['/GUI/Fcount'] = 1
	print "==================================================="
	print "===              GUI ON               ==="
	print "===================================================\n"
	if methods.check('/Analysis/Postsim/Balance'):
		figure(methods['/GUI/Fcount'])
		subplot(221)
		title("E-pop. Current ratio")
		plot(abs(Eib[:,0]),Eib[:,1],"ro")
		ylabel("Inh. Cur")
		xlabel("Exc. Cur")
		subplot(222)
		title("E-pop. Conduct ratio")
		plot(Egb[:,0],Egb[:,1],"bo")
		ylabel("Inh. Cond")
		xlabel("Exc. Cond")
		subplot(223)
		title("I-pop. Current ratio")
		plot(abs(Iib[:,0]),Iib[:,1],"rx")
		ylabel("Inh. Cur")
		xlabel("Exc. Cur")
		subplot(224)
		title("I-pop. Conduct ratio")
		plot(Igb[:,0],Igb[:,1],"bx")
		ylabel("Inh. Cond")
		xlabel("Exc. Cond")
		methods['/GUI/Fcount'] += 1
		
		
		
		#ccnt += 1
		#spc += np.sum(frate[marks[mx-1][0]:marks[mx+1][0]])
	#if ccnt > 0:
		#spc /= ccnt
	#spc,ccnt = 0.,0.

	
	#methods['Analysis/Postsim/E/PeakDetector/Peaks'] = peaks
#DB>>
if methods.check("/GUI"):
	if P_Spectrum or N_Spectrum:
		figure(methods['/GUI/Fcount'])
		methods['/GUI/Fcount'] += 1
		if P_Spectrum and N_Spectrum:
			subt = 120
			subn = 1
		else:
			subt = 110
			subn = 1
		if N_Spectrum:
			subplot(subt + subn)
			subn += 1
			title("Neuronal Sectrum")
			for pop in  methods['/Analysis/Postsim'].dict():
				#if methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum"):
					#if pop == 'E':
						#bar(methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][1:,0],
							#methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][1:,1],0.5,color="k",edgecolor="k")
					#elif pop == 'I':
						#bar(methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][1:,0],
							#methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][1:,1],0.5,color="r",edgecolor="r")
					#else:	   
						#bar(methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][1:,0],
							#methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][1:,1],0.5)
				if methods.check('/Analysis/Postsim/'+pop+"/N_Spectrum"):
					if pop == 'E':
						plot(methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][:,0],
							 methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][:,1],'k-')
					elif pop == 'I':
						plot(methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][:,0],
							 methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][:,1],'r-')
					else:	   
						plot(methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][:,0],
							 methods['/Analysis/Postsim/'+pop+"/N_Spectrum"][:,1],'-')
		if P_Spectrum:
			subplot(subt + subn)
			subn += 1
			title("Population Sectrum")
			for pop in  methods['/Analysis/Postsim'].dict():
				if methods.check('/Analysis/Postsim/'+pop+"/P_Spectrum"):
					#if pop == 'E':
						#bar(methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][1:,0],
								#methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][1:,1],0.5,color="k",edgecolor="k")
					#elif pop == 'I':
						#bar(methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][1:,0],
								#methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][1:,1],0.5,color="r",edgecolor="r")
					#else:	   
						#bar(methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][1:,0],
								#methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][1:,1],0.5)
					if pop == 'E':
						plot(methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][:,0],
							 methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][:,1],"k-")
					elif pop == 'I':
						plot(methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][:,0],
							 methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][:,1],"r-")
					else:	   
						plot(methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][:,0],
							 methods['/Analysis/Postsim/'+pop+"/P_Spectrum"][:,1],"-")

if methods.check("/GUI"):
	def keypass(event):
		global k,p
		if   event.key == "e":
			k -= 1
			if k < 0: k = methods['/Populations/E/n']-1
			Ev.set_ydata(populations['E'][k].rec['v'])
			#Eei.set_ydata(populations['E'][k].rec['eg'])
			#Eii.set_ydata(populations['E'][k].rec['ig'])
			#Eei.set_ydata(populations['E'][k].rec['ei'])
			#Eii.set_ydata(populations['E'][k].rec['ii'])
			#Ei.set_ydata(array(populations['E'][p].rec['ig'])*(-50.-populations['E'][p].isyn.e) + \
				 #array(populations['E'][p].rec['eg'])*(-50.-populations['E'][p].esyn.e) + \
				 #populations['E'][k].rec['i']
			#)
		elif event.key == "E":
			k += 1
			if k >= methods['/Populations/E/n']: k = 0
			Ev.set_ydata(populations['E'][k].rec['v'])
			#Eei.set_ydata(populations['E'][k].rec['eg'])
			#Eii.set_ydata(populations['E'][k].rec['ig'])
			#Eei.set_ydata(populations['E'][k].rec['ei'])
			#Eii.set_ydata(populations['E'][k].rec['ii'])
			#Ei.set_ydata(array(populations['E'][p].rec['ig'])*(-50.-populations['E'][p].isyn.e) + \
				 #array(populations['E'][p].rec['eg'])*(-50.-populations['E'][p].esyn.e) + \
				 #populations['E'][k].rec['i']
			#)
		if   event.key == "i":
			p -= 1
			if p < 0: p = methods['/Populations/I/n']-1
			Iv.set_ydata(populations['I'][p].rec['v'])
			#Iei.set_ydata(populations['I'][p].rec['eg'])
			#Iii.set_ydata(populations['I'][p].rec['ig'])
			#Iei.set_ydata(populations['I'][p].rec['ei'])
			#Iii.set_ydata(populations['I'][p].rec['ii'])
			#Ii.set_ydata(array(populations['I'][p].rec['ig'])*(-50.-populations['I'][p].isyn.e) + \
				 #array(populations['I'][p].rec['eg'])*(-50.-populations['I'][p].esyn.e) + \
				 #populations['I'][k].rec['i']
			#)
		elif event.key == "I":
			p += 1
			if p >= methods['/Populations/I/n']: p = 0
			Iv.set_ydata(populations['I'][p].rec['v'])
			#Iei.set_ydata(populations['I'][p].rec['eg'])
			#Iii.set_ydata(populations['I'][p].rec['ig'])
			#Iei.set_ydata(populations['I'][p].rec['ei'])
			#Iii.set_ydata(populations['I'][p].rec['ii'])
			#Ii.set_ydata(array(populations['I'][p].rec['ig'])*(-50.-populations['I'][p].isyn.e) + \
				 #array(populations['I'][p].rec['eg'])*(-50.-populations['I'][p].esyn.e) + \
				 #populations['I'][k].rec['i']
			#)
		Fig.canvas.draw()
	if methods.check("/GUI/Save"):
		Fig = figure(methods['/GUI/Fcount'],figsize=(31/2.54,24/2.54))
	else:
		Fig = figure(methods['/GUI/Fcount'])
	#ax=subplot(311)
	ax=subplot(511)
	plot(spikes['E'][:,0],spikes['E'][:,1],"k.")
	plot(spikes['I'][:,0],spikes['I'][:,1]+methods["/Populations/E/n"]+5                             ,"r.")
	#plot(spikes['S'][:,0],spikes['S'][:,1]+methods["/Populations/E/n"]+methods["/Populations/I/n"]+10,"g|")
	for k in xrange(methods['/Populations/E/n']):
		if len(where(spikes['E'][:,1]==k)[0]) > 1: break
	for p in xrange(methods['/Populations/I/n']):
		if len(where(spikes['I'][:,1]==p)[0]) > 1: break
	
	#subplot(312, sharex=ax)
	#plot(t,array(populations['E'][k].rec['v'])*0.8,'k-',lw=2)
	#subplot(313, sharex=ax)
	#plot(t,populations['I'][0].rec['v'],'r-',lw=2)

	#subplot(613, sharex=ax)
	#plot(t,populations['E'][k].rec['ei'],"r-")
	#plot(t,populations['E'][k].rec['ii'],"b-")
	#subplot(614, sharex=ax)
	#plot(t,populations['E'][k].rec['ei'],"r-")
	#plot(t,populations['I'][k].rec['ii'],"b-")

	subplot(512, sharex=ax)
	Ev, =plot(t,populations['E'][k].rec['v'],"k-")
	Iv, =plot(t,populations['I'][p].rec['v'],"r-")
	subplot(513, sharex=ax)
	#Eei,=plot(t,populations['E'][k].rec['eg'],"r-")
	#Eii,=plot(t,populations['E'][k].rec['ig'],"b-")
	#Eei,=plot(t,populations['E'][k].rec['ei'],"r-")
	#Eii,=plot(t,populations['E'][k].rec['ii'],"b-")
	Eei = np.zeros(np.array(populations['E'][0].rec['ei']).shape)
	Eii = np.zeros(np.array(populations['E'][0].rec['ii']).shape)
	for en in populations['E']:
		Eei += np.array(en.rec['ei'])
		Eii += np.array(en.rec['ii'])
	plot(t,Eei,"r-")
	plot(t,Eii,"b-")
	#vc = array(populations['E'][p].rec['ig'])*(-50.-populations['E'][p].isyn.e) + \
	     #array(populations['E'][p].rec['eg'])*(-50.-populations['E'][p].esyn.e) + \
	     #populations['E'][k].rec['i']
	#Ei,=plot(t,vc,"k-")
	subplot(514, sharex=ax)
	#Iei,=plot(t,populations['I'][p].rec['eg'],"r-")
	#Iii,=plot(t,populations['I'][p].rec['ig'],"b-")
	#Iei,=plot(t,populations['I'][p].rec['ei'],"r-")
	#Iii,=plot(t,populations['I'][p].rec['ii'],"b-")
	Iei = np.zeros(np.array(populations['I'][0].rec['ei']).shape)
	Iii = np.zeros(np.array(populations['I'][0].rec['ii']).shape)
	for ni in populations['I']:
		Iei += np.array(ni.rec['ei'])
		Iii += np.array(ni.rec['ii'])
	plot(t,Iei,"r-")
	plot(t,Iii,"b-")
	#vc = array(populations['I'][p].rec['ig'])*(-50.-populations['I'][p].isyn.e) + \
	     #array(populations['I'][p].rec['eg'])*(-50.-populations['I'][p].esyn.e) + \
	     #populations['I'][k].rec['i']
	#Ii,=plot(t,vc,"k-")
	subplot(515, sharex=ax)
	if methods.check("/Analysis"):
		plot(arange(0,firingrate['E'].shape[0],1.),firingrate['E'],"k-")
		plot(arange(0,firingrate['I'].shape[0],1.),firingrate['I'],"r-")
		maxfr = max(firingrate['E'])
		if maxfr< max(firingrate['I']):maxfr = max(firingrate['I'])
		ep = array( [ (x,maxfr) for x in methods['/Analysis/Postsim/E/PeakDetector/Peaks'] ] )
		if ep.shape[0] > 5:
			plot(ep[:,0],ep[:,1],"k*")
		ip = array( [ (x,maxfr) for x in methods['/Analysis/Postsim/I/PeakDetector/Peaks'] ] )
		if ip.shape[0] > 5:
			plot(ip[:,0],ip[:,1],"r*")
	Fig.canvas.mpl_connect('key_press_event', keypass)
	#ymin,ymax = ylim()
	#for p in methods[aps+'E'+'/PeakDetector/Peaks']:
		##print 'E',p
		#plot([p,p],[ymax/2,ymax],"r--",lw=3)
	#for p in methods[aps+'I'+'/PeakDetector/Peaks']:
		##print 'I',p
		#plot([p,p],[ymax/2,ymax],"k--",lw=3)
	##for p in methods[aps+'S'+'/PeakDetector/Peaks']:
		##print 'S',p
		##plot([p,p],[ymax/2,ymax],"g--",lw=3)
	#subplot(616, sharex=ax)
	#plot(arange(0,firingrate['S'].shape[0],1.),firingrate['S'],"go")
	##print "SOTH:", methods.check(aps+'E'+'/PeakDetector/SmoothRate')

	##plot(arange(0,methods[aps+'E'+'/PeakDetector/SmoothRate'].shape[0],1.),methods[aps+'E'+'/PeakDetector/SmoothRate'],"ro")
	##plot(arange(0,methods[aps+'I'+'/PeakDetector/SmoothRate'].shape[0],1.),methods[aps+'I'+'/PeakDetector/SmoothRate'],"bo")
	##plot(arange(0,methods[aps+'S'+'/PeakDetector/SmoothRate'].shape[0],1.),methods[aps+'S'+'/PeakDetector/SmoothRate'],"go")
	if methods.check("/GUI/Save"):
		if not type(methods["/GUI/Save"]) is str:
			methods["/GUI/Save"] = methods.simrec["hash"]+".jpg"
		Fig.savefig(methods["/GUI/Save"])
	else:
		show()
#<<DB

if methods.check('/CONFIG/simdb'):
	methods.simdbwrite(methods['/CONFIG/simdb'])
if methods.check('/CONFIG/config'):
	confile = str(methods['/CONFIG/config'])
	methods['/CONFIG/config'] = False
	methods.genconfile(confile)
exit(0)
