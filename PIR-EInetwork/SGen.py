"""
Template for single spike source

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

class Sg:
	def __init__(self, init=[0,0.]):
		self.soma = h.Section() #dummy compartment
		self.output = h.VecStim()
		self.id = init[0]
		self.tS = init[1]
		#Spike Recorder
		self.spks	= h.Vector()
		self.recorder = h.NetCon(self.output,None,sec=self.soma)
		self.recorder.threshold = 0.
		self.recorder.record(self.spks)
	def activate(self,sequence):			
		self.src = h.Vector(sequence.shape[0])
		self.src.from_python(sequence)
		self.output.play(self.src)

if __name__ == "__main__":
	g = Sg()
	g.activate(lambda i,t: t+random.random()*100, 0, 500)
	print array(g.src)

