import os, sys, types, logging, re, fnmatch, hashlib, zlib, simdb
from numpy import *
from textwrap import wrap as tw
from simconfig import getconfig
from datetime import datetime
from random import randint

from datetime import datetime

class methodtree(dict):
	def __init__(self, hsymbol="/"):
		self.hsymbol = hsymbol
		self.namelist = [] 				#Contains only name roots in this tree
		self.objnames = []			    #Contains only fuul names of object not a subtrees
		self.logger = logging.getLogger("simtools.simmethods.methodstree")
	def __setitem__(self, key, value):
		skey = key.lstrip(" \n\t\r/")
		parts = skey.split(self.hsymbol, 1)
		if type(value) is dict:
			if skey in self.objnames: self.objnames.remove(skey)
			for n,v in self.mapdict(value):
				self[skey+self.hsymbol+n]=v
		elif isinstance(value,methodtree): 
			if skey in self.objnames: self.objnames.remove(skey)
			#super(methodtree, self).__setitem__(skey, value)
			for name in value:
				self[skey+name.replace(value.hsymbol,self.hsymbol)] = value[name]
		elif len(parts) == 2:
		#if len(parts) == 2:
			self.__cleanobjnames__(skey+self.hsymbol)
			if skey not in self.objnames: self.objnames.append(skey)
			#>> Problem with ping-bio >>> 
			#if skey in self.objnames: self.objnames.remove(skey)
			#<< Problem with ping-bio <<<
			if not super(methodtree,self).__contains__( parts[0] ): 
				super(methodtree,self).__setitem__(parts[0], methodtree(hsymbol = self.hsymbol))
			elif not isinstance(self[parts[0]], methodtree)       : 
				super(methodtree,self).__setitem__(parts[0], methodtree(hsymbol = self.hsymbol))
			if parts[0] not in self.namelist: self.namelist.append(parts[0])
			self[parts[0]].__setitem__(parts[1], value)
		else:
			self.__cleanobjnames__(skey+self.hsymbol)
			if skey not in self.namelist: self.namelist.append(skey)		
			if skey not in self.objnames: self.objnames.append(skey)
			super(methodtree, self).__setitem__(skey, value)
		#DB>>
		#self.logger.debug("  > MT > {}:{}".format(skey, self.objnames))
		#<<DB

	def __getitem__(self, key):
		if key is None: return self.dict()
		skey = key.lstrip(" \n\t\r/")
		parts = skey.split(self.hsymbol, 1)
		if len(parts) == 2:
			if not super(methodtree,self).__contains__( parts[0] ): 
				self.logger.error("Cannot find a subtree {} in the tree".format(parts[0]))
				raise KeyError(   "Cannot find a subtree {} in the tree".format(parts[0]))
			try:
				return self[parts[0]][parts[1]]
			except BaseException as e:
				self.logger.error("Cannot resolve name {} in {} : {}".format(parts[1],parts[0],e))
				raise KeyError(   "Cannot resolve name {} in {} : {}".format(parts[1],parts[0],e))
		else:
			if not super(methodtree,self).__contains__( skey ): 
				self.logger.error("Cannot find an item \'{}\' in the tree ".format(skey))
				raise KeyError(   "Cannot find an item \'{}\' in the tree ".format(skey))
			return super(methodtree, self).__getitem__(skey)
	def __contains__(self,key):
		#if key[0] == self.hsymbol : key = key[1:]
		skey = key.lstrip(" \n\t\r/")
		parts = skey.split(self.hsymbol, 1)
		if len(parts) == 2:
			if not super(methodtree, self).__contains__(parts[0]): return False
			if not isinstance(self[parts[0]], methodtree) : return False
			return parts[1] in self[parts[0]]
		else:
			if not super(methodtree, self).__contains__(skey): return False
			return True
	def __cleanobjnames__(self,key):
		for objname in filter(lambda x: x[:len(key)] == key, self.objnames):
			self.objnames.remove(objname)
	def __delitem__(self, key):
		#if key[0] == self.hsymbol : key = key[1:]
		skey = key.lstrip(" \n\t\r/")
		parts = skey.split(self.hsymbol, 1)
		if len(parts) == 2:
			if parts[0] not in self: 
				self.logger.error("Cannot find a subtree {} in the tree".format(parts[0]))
				raise KeyError(   "Cannot find a subtree {} in the tree".format(parts[0]))
			self[parts[0]].__delitem__(parts[1])
		else:
			if skey not in self: 
				self.logger.error("Cannot find an item {} in the tree".format(skey))
				raise KeyError(   "Cannot find an item {} in the tree".format(skey))
			super(methodtree,self).__delitem__(skey)
			if skey not in self.namelist: 
				self.logger.error("Cannot find an item {} in the namelist".format(skey))
				raise KeyError(   "Cannot find an item {} in the namelist".format(skey))
			self.namelist.remove(skey)
		self.__cleanobjnames__(skey)
		
	def keys(self,parent=""): return list(iter(self))

	def __iter__(self, parent=""):
		for name in self.objnames.__iter__(): yield self.hsymbol+name
	def dict(self):
		for name in self.namelist.__iter__(): yield name
	def check(self, key):
		if not self.__contains__(key): return False
		xvalue= self[ key ]
		if type( xvalue ) is bool or type( xvalue ) is int: return bool( xvalue )
		elif xvalue is None :  return False
		elif type( xvalue ) is str and xvalue == '': return False
		elif (type( xvalue ) is list or type( xvalue ) is tuple) and len(xvalue) == 0: return False
		else: return True
	def mapdict(self,mapd,parent=""):
		d = []
		for n in mapd:
			if type(mapd[n]) is dict:
				d += self.mapdict(mapd[n], parent = parent+self.hsymbol+n )
			else:
				d.append( (parent+self.hsymbol+n, mapd[n]) )
		return d
	def mapnames(self,parent = ''):
		root = {}
		for name in self.namelist:
			if isinstance(self[name],methodtree):
				root[name]=self[name].mapnames(parent = parent+self.hsymbol+name)
			else:
				root[name]=parent+self.hsymbol+name
		return root
	def printnames(self, space = "", parent=""):
		root = []
		for nidx,name in enumerate( self.namelist ):
			if isinstance(self[name],methodtree):
				lastflag = nidx==len(self.namelist)-1
				root.append( ("%s%sv: %s \n"%(space, "`-" if lastflag else "|-", name), None) )
				root += self[name].printnames(space=space+"  " if lastflag else space+"| ",parent=parent+self.hsymbol+name)
			else:
				root.append( ("%s%s> %s "%(space,"`-" if nidx==len(self)-1 else "|-", name), parent+self.hsymbol+name) ) 
		return root
	#def __repr__(self, space = ""):
		#prn = ""
		#for nidx,name in enumerate( sorted(self) ):
			#if isinstance(self[name],methodtree):
				#rep = "%s%sv: %s "%(space, "`-" if nidx==len(self)-1 else "|-", name)
				#prn += rep+"\n"
				#prn += self[name].__repr__(space=space+"| " if len(self) > 1 else space+"  ")
			#else:
				#rep = "%s%s> %s "%(space,"`-" if nidx==len(self)-1 else "|-", name)
				#if len(rep) < 31:
					#for x in xrange(31-len(rep)):rep += " "
				#if type(self[name]) is str:
					#prn += rep + " : \"{}\"\n".format(self[name])
				#else:
					#prn += rep + " : {}\n".format(self[name])	
		#return prn
	def __str__(self,parent=None):
		#if parent is None: paret = self.hsymbol
		return super(methodtree, self).__str__()
	def exp(self):
		return [ (name,self[name]) for name in self.objnames ]
	#def exp(self):
		#return [ (name,self[name]) for name in self ]
	def imp(self, data):
		if not type(data) is list:
			self.logger.error("Can import data only from list of tuples, {} given".format(type(data)) )
			raise TypeError(  "Can import data only from list of tuples, {} given".format(type(data)) )
		if len(data) == 0:
			self.logger.error("Empty data for importing")
			raise ValueError( "Empty data for importing")
		for idx, d in enumerate(data):
			if not type(d) is tuple:
				self.logger.error("Record {} for importing data is not a tuple".format(idx))
				raise TypeError( "Record {} for importing data is not a tuple".format(idx))
			if len(d) != 2:
				self.logger.error("Record {} for importing data doesn't have two fields (name and value): given {}".format(idx,d))
				raise ValueError( "Record {} for importing data doesn't have two fields (name and value): given {}".format(idx,d))
			name, value = d
			self[name] = value
		return self
			


class simmethods():
	"""
	Class **simmethods** hold parameters record and allows to reset 
	 parameters from config-like file, simulation DB record (simbd) or from command line arguments.
	Methods are peers of name and value separated by = (or set another symbol by separator)
	Names are hierarchically organized in tree.
	Hierarchical position of names are given by standard form /top-level/bottom-level.
	Hierarchical symbol / can be substitute by another one by hsymbol parameter
	
	It gets a default values as a list with tupels of record and
	 annotation.
	Then it can par command line arguments and/or files to alter 
	 parameters and add new.
	
	You need create a object of defaultgen,
	then call setdefault function for set of parameters,
	and then call generate to get a dictionary.
	
	If user pass -h for you program you want to call
	printhelp function for annotating help of DEFAULT arguments.
	"""
	is_lambda = lambda self, value    : isinstance(value, types.LambdaType) and value.__name__ == '<lambda>'
	def __init__(self, 
		default=None, presets=None, argvs=None,
		target=None,  localcontext=None,
		vseparator="=",hsymbol="/",refsymbol="@", strefsymbol="$", refhash="#", rseparator=";",
		filersh='-f', filerln='--file', simdbsh='-s', simdbln='--simdb', simdbsp=':',
		pure=False):
		"""
		create a object with optional parameters
		@opt   presets     - this set BEFORE default parameters, but will no indicated in help
		                   - very useful for seting up MPI rank and so on 
		@opt   default     - list of default parameters
		                   - each parameter is a tuple of two strings: parameter setting and parameter annotation
		                   - EXAMPLE: default= [ ('/A=2','Parameter A'), ('/B=3', 'Parameter B') ]
		@opt   argvs       - command line arguments, which will be processed after defaults.
		                   - can contain
		---
		@opt   target      - name of simmethods object which will be created by this constructor
		@opt   localcontext- local name space where all exec should be run.
		                   - can be alter by localcontext parameter in generate function
		                   - if None, simmethods uses current context from globals()

		@opt   vseparator  - a symbol to separate a parameter name from a value (default =)
		@opt   hsymbol     - a symbol to present hierarchical position within the parameter name (default /)
		@opt   refsymbol   - a symbol to indicate a referenced name (default @)
		@opt   strefsymbol - a symbol to indicate a referenced name if the value should be converted back to a string (default $)
		@opt   refhash     - a symbol to indicate a referenced name if the value should be a hash sum of the content (default #)
		@opt   rseparator  - a symbol to separate parameters recods in simdb (default ;)
		@opt   pure        - if True parameters with wrong format will raise an exception (default False).
		@opt   filersh     - a short key for read from file (default "-f")
		@opt   filerln     - a long  key for read from file (default "--file")
		@opt   simdbsh     - a short key for read from simdb (default "-s")
		@opt   simdbln     - a long  key for read from simdb (default "--simdb")
		@opt   simdbsp     - a separator between simdb filename and key-filed and many name-filters (default ":")

		"""
		self.vseparator  = vseparator
		self.hsymbol     = hsymbol
		self.refsymbol   = refsymbol
		self.strefsymbol = strefsymbol
		self.refhash     = refhash
		self.rseparator  = rseparator
		self.pure        = pure
		self.filersh     = filersh
		self.filerln     = filerln
		self.simdbsh     = simdbsh
		self.simdbln     = simdbln
		self.simdbsp     = simdbsp
		
		self.namespace   = methodtree()
		self.hashspace   = methodtree()
		self.methods     = methodtree()
		self.dtarget     = target
		self.dlocalctx   = localcontext
		
		if type(default) is str:
			self.default = self.str2default(default)
		else:
			self.default = default
		
		self.logger = logging.getLogger("simtools.simmethods")
		self.simrec = None

		if type(presets) is list or type(presets) is tuple:
			self.updatenamespace ( presets  )
		elif type(presets) is str:
			self.updatenamespace ( [ presets ] )
		if not default is None:
			self.updatenamespace ( [ var for var,_ in self.default ] )
		if not argvs is None:
			self.readargvs(argvs)

	def __setitem__(self, key, value): self.methods.__setitem__(key, value)
	def __getitem__(self, key)       : 
		if key[0] == "#":
			return self.gethash(key[1:])
		return self.methods.__getitem__(key)
	def __contains__(self,key)       : return self.methods.__contains__(key)
	def __delitem__(self, key)       : self.methods.__delitem__(key)
	def __iter__(self):
		for name in self.methods.__iter__(): yield name
	def check(self, key)	         : return self.methods.check(key)
	def dict(self):
		for name in self.methods.dict():yield name
	def gethash(self, name=''):
		if not name in self.methods: return None
		if isinstance(self.methods[name], methodtree):
			achash=""
			for n in self.methods[name].dict():
				achash += ":"+self.gethash(name+self.hsymbol+n)
			return achash[1:]
		elif name in self.hashspace:
			return self.hashspace[name]
		elif self.is_lambda(self[name]):
			if name in self.namespace:
				self.hashspace[name] = hashlib.sha1(self.namespace).hexdigest()
				return self.hashspace[name]
			else:
				self.logger.warning(" > LAMBDA function not in namespace")
				self.hashspace[name] = hashlib.sha1(str(self.methods[name])).hexdigest()
				return self.hashspace[name]
		else:
			self.hashspace[name] = hashlib.sha1(str(self.methods[name])).hexdigest()
			return self.hashspace[name]
		return None
	def getrecord(self, record):
		"""
		@function **methods.getrecord** read parameter setting and select name of parameter and value
		@param record      - one parameter record.
		@returns 
			name and value
		"""
		record = record.strip(' \t\n\r')
		try:
			name,value = record.split(self.vseparator,1)
		except:
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.getrecord)")
			self.logger.error("		     : Cannot separate name and value of a record {} by separator {}".format(record,self.vseparator))
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Cannot separate name and value of a record {} by separator {}".format(record,self.vseparator))

			return None,None
		try:
			name,value = name.rstrip(' \t\n\r/'), value.strip(' \t\n\r')
		except:
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.getrecord)")
			self.logger.error("		     : Cannot strip empty lines from name and value of a record {} ".format(record))
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Cannot strip empty lines from name and value of a record {} ".format(record))			
			return None,None
		return name,value

	def update(self,*prms):
		self.updatenamespace([prm for prm in prms])
		self.generate()
	def updatenamespace(self,argv):
		for arg in argv:
			iname,value = self.getrecord(arg)
			#DB>>
			#self.logger.debug("UN > {}={}".format(iname,value))
			#<<DB
			if iname is None or value is None: continue
			self.namespace[iname] = value
			self.hashspace[iname] = hashlib.sha1(value).hexdigest()
	
	def readargvs(self, argv, separator=None, hsymbol=None, refsymbol=None, strefsymbol=None, refhash=None, pure=None,\
						filersh=None, filerln=None, simdbsh=None, simdbln=None, simdbsp=None\
		):
		"""
		@function simmethods.readargvs strip and reads list of program arguments.
		@param argv        - list of program arguments
		--- you can alter the separators for reading
		@opt   separator   - a symbol to separate a parameter name from a value (default =)
		@opt   hsymbol     - a symbol to present hierarchical position within the parameter name (default /)
		@opt   refsymbol   - a symbol to indicate a referenced name (default @)
		@opt   strefsymbol - a symbol to indicate a referenced name if the value should be converted back to a string (default $)
		@opt   refhash     - a symbol to indicate a referenced name if the value should be a hash sum of the content (default #)
		@opt   rseparator  - a symbol to separate parameters recods in simdb (default ;)
		@opt   pure        - if True parameters with wrong format will raise an exception (default False).
		@opt   filersh     - a short key for read from file (default "-f")
		@opt   filerln     - a long  key for read from file (default "--file")
		@opt   simdbsh     - a short key for read from simdb (default "-s")
		@opt   simdbln     - a long  key for read from simdb (default "--simdb")
		@opt   simdbsp     - a separator between simdb filename and key-filed and many name-filters (default ":")
		"""
		tempevn = self.vseparator, self.hsymbol, self.refsymbol, self.refhash, self.strefsymbol, self.pure,\
		          self.filersh, self.filerln, self.simdbsh, self.simdbln, self.simdbsp
		if  not separator   is None : self.vseparator  = separator
		if  not hsymbol     is None : self.hsymbol     = hsymbol
		if  not refsymbol   is None : self.refsymbol   = refsymbol
		if  not strefsymbol is None : self.strefsymbol = strefsymbol
		if  not refhash     is None : self.refhash     = refhash
		if  not pure        is None : self.pure        = pure
		if  not filersh is None     : self.filersh     = filersh
		if  not filerln is None     : self.filerln     = filerln
		if  not simdbsh is None     : self.simdbsh     = simdbsh
		if  not simdbln is None     : self.simdbln     = simdbln
		if  not simdbsp is None     : self.simdbsp     = simdbsp
		#DB>>
		#print self.separator, self.hsymbol, self.refsymbol, self.strefsymbol, self.pure, self.filersh, self.filerln, self.simdbsh, self.simdbln, self.simdbsp
		#<<DB

		rargv,skiper = [], False
		for iarg, arg in enumerate(argv):
			if skiper:
				skiper = False
				continue
			arg = arg.strip(" \t\n\r")
			if not self.filersh is None or not self.filerln is None:
				if arg[:len(self.filersh)] == self.filersh:
					if len(arg) > len(self.filersh):
						if arg[len(self.filersh)] == "=":
							filename = arg[len(self.filersh)+1:]
						else:
							filename = arg[len(self.filersh):]
					else:
						filename =argv[iarg+1]
						skiper = True

					self.readfile(filename)
					continue
				if arg[:len(self.filerln)] == self.filerln:
					if len(arg) > len(self.filerln):
						if arg[len(self.filerln)] == "=":
							filename = arg[len(self.filerln)+1:]
						else:
							filename = arg[len(self.filerln):]
					else:
						filename =argv[iarg+1]
						skiper = True
					self.readfile(filename)
					continue
			if not self.simdbsh is None or not self.simdbln is None:
				if arg[:len(self.simdbsh)] == self.simdbsh:
					if len(arg) > len(self.simdbsh):
						if arg[len(self.simdbsh)] == "=":
							filename = arg[len(self.simdbsh)+1:]
						else:
							filename = arg[len(self.simdbsh):]
					else:
						filename =argv[iarg+1]
						skiper = True
					self.readb(filename)
					continue
				if arg[:len(self.simdbln)] == self.simdbln:
					if len(arg) > len(self.simdbln):
						if arg[len(self.simdbln)] == "=":
							filename = arg[len(self.simdbln)+1:]
						else:
							filename = arg[len(self.simdbln):]
					else:
						filename =argv[iarg+1]
						skiper = True
					self.readb(filename)
					continue
			rargv.append(arg)
		self.updatenamespace(rargv)
		self.vseparator, self.hsymbol, self.refsymbol, self.refhash, self.strefsymbol, self.pure,\
		          self.filersh, self.filerln, self.simdbsh, self.simdbln, self.simdbsp = tempevn

	def readfile(self, infile, separator=None, hsymbol=None,refsymbol=None, strefsymbol=None, refhash=None, pure=None):
		"""
		@function methods.readfile strip and reads list of program arguments.
		@param infile       - input file name or file object
		--- you can alter key for separtors for reading
		@opt   separator   - a symbol to separate a parameter name from a value (default =)
		@opt   hsymbol     - a symbol to present hierarchical position within the parameter name (default /)
		@opt   refsymbol   - a symbol to indicate a referenced name (default @)
		@opt   strefsymbol - a symbol to indicate a referenced name if the value should be converted back to a string (default $)
		@opt   refhash     - a symbol to indicate a referenced name if the value should be a hash sum of the content (default #)
		@opt   pure        - if True parameters with wrong format will raise an exception (default False).

		"""
		tempevn = self.vseparator, self.hsymbol, self.refsymbol, self.strefsymbol, self.refhash, self.pure
		if  not separator is None   : self.vseparator  = separator
		if  not hsymbol is None     : self.hsymbol     = hsymbol
		if  not refsymbol is None   : self.refsymbol   = refsymbol
		if  not strefsymbol is None : self.strefsymbol = strefsymbol
		if  not refhash     is None : self.refhash     = refhash
		if  not pure is None        : self.pure        = pure
		self.logger.info("=====================================================")
		self.logger.info("=== READ FILE {} ====".format(infile))
		if type(infile) is not str and type(infile) is not file:
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readfile)")
			self.logger.error("		     : Wrong type of input file. Should be str or file but given {}".format(type(infile)))
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Wrong type of input file. Should be str or file but given {}".format(type(infile)))
			return True
		if type(infile) is str : 
			if not os.access(infile,os.R_OK):
				self.logger.error("----------------------------------------------------")
				self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readfile)")
				self.logger.error("		     : Cannot access file \'{}\'".format(infile))
				self.logger.error("----------------------------------------------------")		
				if self.pure                 : raise ValueError("Cannot access file \'{}\'".format(infile))
				return True
			try:
				infile = open(infile, "r")
			except:
				self.logger.error("----------------------------------------------------")
				self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readfile)")
				self.logger.error("		     : Cannot open file \'{}\'".format(infile))
				self.logger.error("----------------------------------------------------")		
				if self.pure                 : raise ValueError("Cannot open file \'{}\'".format(infile))
				return True
		argv = []
		for lst in infile.readlines():
			if lst[0] == "\n" or lst[0] == "#": continue
		
		
			if lst[0] != " " and lst[0] != "\t" and lst[0] != "\n" and lst[0] != "\r": 
				argv.append(lst.strip(' \t\n\r').split("#",1)[0].split(";",1)[0])
				#DB>>
				#self.logger.debug("FS > {}".format(argv[-1]))
				#<<DB
			else:
				if len(argv) < 1:
					self.logger.error("----------------------------------------------------")
					self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readfile)")
					self.logger.error("		     : First argument starts not blank symbol\'{}\'".format(lst))
					self.logger.error("----------------------------------------------------")		
					if self.pure             : raise ValueError("First argument starts not blank symbol\'{}\'".format(lst))
					return True
				argv[-1] += lst.strip(' \t\n\r').split("#",1)[0].split(";",1)[0]
		self.updatenamespace(argv)
		self.vseparator, self.hsymbol, self.refsymbol, self.strefsymbol, self.refhash, self.pure = tempevn
		self.logger.info("=====================================================")
		return False
	def readb(self, infile, separator=None, hsymbol=None,refsymbol=None, strefsymbol=None, refhash=None, pure=None, simdbsp=None):
		"""
		@function methods.readb reads parameters from simdb.
		@param infile       - simdb file name:number of record OR hash OR time stamp [:parameter [:parameter...]] 
		--- you can alter key for separators for reading
		@opt   separator  - a symbol to separate a parameter name from a value (default =)
		@opt   hsymbol     - a symbol to present hierarchical position within the parameter name (default /)
		@opt   refsymbol   - a symbol to indicate a referenced name (default @)
		@opt   strefsymbol - a symbol to indicate a referenced name if the value should be converted back to a string (default $)
		@opt   refhash     - a symbol to indicate a referenced name if the value should be a hash sum of the content (default #)
		@opt   pure        - if True parameters with wrong format will raise an exception (default False).
		@opt   simdbsp     - a separator between simdb filename and key-filed and many name-filters (default ":")
		"""
		tempevn = self.vseparator, self.hsymbol, self.refsymbol, self.strefsymbol, self.refhash, self.pure, self.simdbsp
		if  not separator is None   : self.vseparator  = separator
		if  not hsymbol is None     : self.hsymbol     = hsymbol
		if  not refsymbol is None   : self.refsymbol   = refsymbol
		if  not strefsymbol is None : self.strefsymbol = strefsymbol
		if  not refhash     is None : self.refhash     = refhash
		if  not pure is None        : self.pure        = pure
		if  not simdbsp is None     : self.simdbsp     = simdbsp
		if type(infile) is not str:
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readb)")
			self.logger.error("		     : Wrong type of input simdb should be a string but given {}".format(type(infile)))
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Wrong type of input simdb should be a string but given {}".format(type(infile)))
			return True
		fields = infile.split(self.simdbsp)
		
		if not os.access(fields[0],os.R_OK):
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readb)")
			self.logger.error("		     : Cannot access file \'{}\'".format(fields[0]))
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Cannot access file \'{}\'".format(fields[0]))
			return True
		sdb = simdb.simdb(fields[0])
		reckey = 0
		if len(fields) > 1:
			if fields[1] != "":
				try:
					exec "reckey ="+fields[1]
				except:
					try:
						exec "reckey = \'{}\'".format(fields[1])
					except:
						self.logger.error("----------------------------------------------------")
						self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readb)")
						self.logger.error("		     : Cannot convert key {} in simdb \'{}\'".format(fields[1],fields[0]))
						self.logger.error("----------------------------------------------------")		
						if self.pure             : raise ValueError("Cannot convert key {} in simdb \'{}\'".format(fields[1],fields[0]))
						return True
		
		rectype,timestamp,hashstr,message,arglst = sdb.readrec(reckey, lst=True)
		self.logger.info(" v READING SIMDB RECORD " )
		self.logger.info(" |-> TYPE          : {}".format(rectype) )
		self.logger.info(" |-> TIME STAMP    : {}".format(simdb.timestamp2str(timestamp)) )
		self.logger.info(" |-> HASH          : {}".format(hashstr) )
		self.logger.info(" |-> MESSAGE       : {}".format(message) )
		if len(fields) > 2:
			self.logger.info(" |-> NAME FILTERS  : {}".format(fields[2:]) )
		if rectype != "SIMREC":
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.readb)")
			self.logger.error("		     : Cannot read {} type of records in simdb \'{}\'".format(rectype,fields[0]))
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Cannot read {} type of records in simdb \'{}\'".format(rectype,fields[0]))
			return True
		
		if len(fields) > 2:
			argv = []
			for arg in arglst:
				name,value = self.getrecord(arg)
				flag = False
				for opt in fields[2:]:
					flag = flag or fnmatch.fnmatchcase(name, opt)
				if flag: argv.append(arg)
						
		else:
			argv = arglst
		#DB>>
		#with open("x.err","w") as f:
			#for r in argv:
				#f.write(r+"\n")
		#<<DB
		self.updatenamespace(argv)
		self.vseparator, self.hsymbol, self.refsymbol, self.strefsymbol, self.refhash, self.pure, self.simdbsp = tempevn
	def str2default(self, default):
		"""
		@function str2default converts string to list of defaults.
		It uses the same symbol(s) as in simbd, for separation parameter set and annotation
		\ - symbol can be used to expand single lines to several lines. annotation will be expended too.
		"""
		ret,cmd,msg = [], "", ""
		for lines in default.split("\n"):
			work = lines.split(self.rseparator)
			if len(work) < 2 or work[0] == "": continue
			if len(work) > 2: work[1] =" ".join(work[1:])
			work[0] = work[0].strip(" \t\n\r")
			if work[0] == "":
				if cmd != "":
					ret.append( (cmd,msg+work[1]) )
					cmd,msg = "", ""
				continue
			if work[0][-1] == '\\':
				cmd += work[0][:-1]
				msg += work[1]
			else:
				ret.append( (cmd+work[0],msg+work[1]) )
				cmd,msg = "", ""
		#DB>>
		#print default
		#print
		#for x in ret:
			#print x
		#exit(0)
		#<<DB
		return ret
	def builder(self, tree, target, item, delimiter):
		"""
		Finds and resolves build expressions for links and strings
		"""
		if delimiter not in item:
			return item
		result = ''
		copir = item.split(delimiter)
		#BeetDemGuise sugessted to use zip instead map 
		# (see http://codereview.stackexchange.com/questions/52729/configuration-file-with-python-functionality).
		# But zip function returns shortest argument 
		# sequence. So, the 'tail' of item after last
		# delimiter just disappiers in the result and raise
		# an error. Any ideas?
		for prefix,var in map(None,copir[::2],copir[1::2]):
			if prefix is not None: result += prefix
			if var is None: continue
			lmdcheck = self.is_lambda(tree[var])
			if lmdcheck or delimiter is self.refsymbol:
				result += target+".methods[\""+var+"\"]"
			elif delimiter is self.strefsymbol:
				if var in self.namespace:
					result +=  self.namespace[var]
				else:
					result += str(tree[var])
			elif delimiter is self.refhash:
				result += "\'\\\'{}\\\'\'".format(self.gethash(var))
			else:
				return None
			self.dependences.append(var)
		return result

	def resolve_name(self, tree, target, item):
		"""
		Resolves links and string in RHS of parameters
		"""
		# Resolve links First
		result = ''
		bld = self.builder(tree, target, item ,  self.refsymbol)
		if bld is None:	return None
		result = bld
		# then Resolve strings
		bld = self.builder(tree, target, result, self.strefsymbol)
		if bld is None:	return None
		result = bld
		# and then Resolve hashs
		bld = self.builder(tree, target, result, self.refhash)
		if bld is None:	return None
		result = bld
		return unicode(result)

	def generate(self, target=None, localcontext = None, only=None, prohibit=None, text=False):
		"""
		@function generate generate recods in root initial directory
		@opt   target      - string of variable name which will be generate (need for lambda(s)), 
		@opt   localcontext- local name space, usually = globals() or locals()
		@opt   only        - list of names which should be generated, 
		                     may be a regular expression
		@opt   prohibit    - list of name which should be generated
		@opt   text        - bool If true populate tree by strings not actual values
		"""
		if target is None:
			target = self.dtarget
		
		if not localcontext is None:
			self.localcontext = localcontext
		elif not self.dlocalctx is None:
			self.localcontext = self.dlocalctx
		else:
			self.localcontext = globals()
			
		if not target in self.localcontext:			
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.generate)")
			self.logger.error("		     : Target object \'{}\' is not found in context".format(target))
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Target object \'{}\' is not found in context".format(target))
			return None


		#process = list(self.allsets)
		if not only is None:
			if type(only) is str: only = [ only ]

		if not prohibit is None:
			if type(prohibit) is str: prohibit = [ prohibit ]
		

		for name in self.namespace:
			##DB>>
			#print "\nDB>>",name
			#if not only is None:
				#print "DB>>",name,only, reduce(lambda x,y: x or fnmatch.fnmatchcase(name,y), only    , False),"\n"
			##<<DB
			if not only     is None and not reduce(lambda x,y: x or fnmatch.fnmatchcase(name,y), only    , False) : continue
			if not prohibit is None and     reduce(lambda x,y: x or fnmatch.fnmatchcase(name,y), prohibit, False) : continue
			value = self.namespace[name]
			
			self.dependences = []
			if text:				
				if not self.hashspace.check(name):
					self.hashspace[name] = hashlib.sha1(value).hexdigest()
				try:
					exec "{}.methods[\'{}\']=\"{}\"".format(target,name,re.sub(r"\\", "\\\\", re.sub(r"\"","\\\"", re.sub("\'","\\\'", value) ) )) in self.localcontext
				except BaseException as e:
					self.logger.error("----------------------------------------------------")
					self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.generate)")
					self.logger.error("		     : Cannot execute operation {}[\'{}\']=\"{}\": {}".format(
						target, name,
						re.sub(r"\\", "\\\\", re.sub(r"\"","\\\"", re.sub("\'","\\\'", value))),e))
					self.logger.error("----------------------------------------------------")		
					if self.pure             : raise ValueError("Cannot execute operation {}[\'{}\']=\"{}\": {}".format(target,name,re.sub(r"\\", "\\\\", re.sub(r"\"","\\\"", re.sub("\'","\\\'", value))),e))
					return True
			else:
				value = self.resolve_name(self.localcontext[target].methods, target,value)	
				try:
					exec "{}.methods[\'{}\']={}".format(target,name,value) in self.localcontext
				except BaseException as e:
					self.logger.error("----------------------------------------------------")
					self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.generate)")
					self.logger.error("		     : Cannot execute operation {}[\'{}\']={}: {}".format(target,name,value,e))
					self.logger.error("----------------------------------------------------")		
					if self.pure             : raise ValueError("Cannot execute operation {}[\'{}\']={}: {}".format(target,name,value,e))
					return True
				#if not self.hashspace.check(name):
					#self.hashspace[name] = self.gethash(name)
				#self.hashspace[name] = self.gethash(name)
				## Update hash everytime then parameter change 
				## NOTE: it doesn't help with data set up outside simmethods
				self.hashspace[name] = self.gethash(name)#hashlib.sha1(str(self.methods[name])).hexdigest()
				for dep in self.dependences:
					self.hashspace[name] += ":"+self.gethash(dep)
			self.dependences = []
			self.logger.debug( " > % 76s : OK"%(name))
		return False
	def printhelp(self):
		ret = "\n**** PARAMETERS ****\n\n"
		temp = self.namespace
		self.namespace = {}
		self.updatenamespace ( [ name for name,_ in self.default ] )
		#try:
			#genroot = self.generate(traget="genroot")
		
		#if genroot is None:
			#self.logger.debug( " > % 41s : OK"%(name))
			##raise ValueError("Cannot generate help: Default parameters has errors.")
			#ValueError("Cannot generate help: Default parameters has errors.")
		
		for r,c in self.default:
			name,val = self.getrecord(r)
			ret += name + " = "+val+ "\n"
			ret += "\n\t".join( tw(c,width=90,initial_indent="\t       -  ") )
			ret += "\n\n"
		ret += "\n"
		self.namespace = temp
		return ret
	def gendbrecord(self, hsymbol = None, separator=None, unresolved = False):
		"""
		@function gendbrecord generates record for simdb from (!!!) generated methods tree
		@option  hsymbol     - symbol to present hierarchical position in parameter name (default as self.rseparator) 
		@option  separator   - symbol to separate parameters in record (default as self.hsymbol)
		@option  unresolved  - Boolean variable, if True parameters will be pool from pregenerator, not actual values (default False).
		"""
		if hsymbol   is None: hsymbol   = self.hsymbol
		if separator is None: separator = self.rseparator
		def normobj(obj):
			if type(obj) is list:
				return "["+",".join([ normobj(o) for o in obj ])+']'
			elif type(obj) is tuple:
				return "("+",".join([ normobj(o) for o in obj ])+')'
			elif type(obj) is dict:
				return "{"+",".join([ n+":"+normobj(obj[n]) for n in obj ]) +"}"
			elif isinstance(obj, ndarray):
				return "array({})".format([ x for x in obj ])
			elif type(self[name]) is str:
				return "\"{}\"".format( re.sub(r"\\", "\\\\", re.sub(r"\"","\\\"", re.sub("\'","\\\'", obj))) )
			else:
				return "{}".format(obj)
			
		setnames, rec, lhash = [], "", ""
		for name in self:
			if self.is_lambda(self[name]):
				if name in self.namespace:
					rec += separator+"{}={}".format(
						name.replace(self.hsymbol,hsymbol),
						self.namespace[name]
					)
				else:
					rec += separator+"{}=\'{}\'".format(
						name.replace(self.hsymbol,hsymbol),
						self[name]
					)
				lhash += self["#"+name]
			elif isinstance(self[name], ndarray):
				rec+= separator+"{}=array({})".format(
					name.replace(self.hsymbol,hsymbol),
					[ x for x in self[name] ]
				)
				lhash += self["#"+name]
			#elif isinstance(x_tree[name], methodtree):
				##DB>>
				##print name, x_tree[name]
				##print "^Xv"
				##<<DB
				#subrec, subhash = self.gendbrecord(
					#hsymbol    = hsymbol, 
					#separator  = separator, 
					#unresolved = unresolved, 
					#x_tree     = x_tree[name],
					#x_prefix   = "%s"%(x_prefix+hsymbol+name  if x_prefix != "" else name)
				#)
				#rec   += subrec
				#lhash += subhash
				##DB>>
				##exit(0)
				##<<DB
			elif type(self[name]) is str:
				rec += separator+"{}=\"{}\"".format(
					name.replace(self.hsymbol,hsymbol),
					re.sub(r"\\", "\\\\", re.sub(r"\"","\\\"", re.sub("\'","\\\'", self[name]))) 
					)
				lhash += self["#"+name]
				#lhash += self["#"+x_prefix.replace(hsymbol,self.hsymbol)+self.hsymbol+name]
				#continue
			elif unresolved and name in self.namespace:
				rec += separator+"{}={}".format(
					name.replace(self.hsymbol,hsymbol),
					re.sub(r"\\", "\\\\", re.sub(r"\"","\\\"", re.sub("\'","\\\'", self.namespace[name]))) 
				)
				lhash += self["#"+name]
				#lhash += self["#"+x_prefix.replace(hsymbol,self.hsymbol)+self.hsymbol+name]
			else:
				rec += separator+"{}=".format(name.replace(self.hsymbol,hsymbol))+normobj(self[name])
				#DB>>
				#print "XDBX", name, type(name), self["#"+name], type(self["#"+name])
				#<<DB
				lhash += self["#"+name]
				#lhash += self["#"+x_prefix.replace(hsymbol,self.hsymbol)+self.hsymbol+name]
				
		return rec.replace("\n","")[1:],lhash
	def simdbrecord(self, message = None, timestamp = None, rechash = True, zipped = True, hsymbol = None, separator = None, unresolved = False):
		"""
		@function simdbrecord prepare record for simdb from again (!!!) generated methods tree
		@option  message     - message for record. If None editor will be evoked to provide message
		@option  timestamp   - fix moment when record made. in None - now
		@option  rechash     - Boolean variable, if True hash will be calculate from resulted record and
		                       not recurrent hash with all dependences. Last one maybe very long! (default True)
		@option  zipped      - Boolean variable, if True record will be zipped and SIMZIP instead of SIMREC will record (default True)
		@option  hsymbol     - symbol to present hierarchical position in parameter name (default as self.rseparator) 
		@option  separator   - symbol to separate parameters in record (default as self.hsymbol)
		@option  unresolved  - Boolean variable, if True parameters will be pool from pregenerator, not actual values (default False).
		"""
		if hsymbol   is None: hsymbol   = self.hsymbol
		if separator is None: separator = self.rseparator
		if not self.simrec is None: del self.simrec
		self.simrec = {}
		rec,hashline = self.gendbrecord(hsymbol=hsymbol, separator=separator, unresolved=unresolved)
		if zipped:
			self.simrec["rectype"] = "SIMZIP"
			self.simrec["record" ] = zlib.compress(rec,9).encode("base64").replace('\n', '')
		else:
			self.simrec["rectype"] = "SIMREC"
			self.simrec["record" ] = rec
		if rechash:	hashline = hashlib.sha1(rec).hexdigest()
		self.simrec["hash"     ] = hashline
		if timestamp is None:
			now = datetime.now()
			self.simrec["timestamp"] = (now.year, now.month, now.day, now.hour, now.minute, now.second, randint(0,999))
		else:
			self.simrec["timestamp"] = timestamp
		self.simrec["message"  ] = message
		self.simrec["separator"] = separator
		
		if message is None:
			self.simrec['messagefile'] = hashline+".simbd-message"
			with open(self.simrec['messagefile'],"w") as fd:
				fd.write("\n\n#HASH:{} TIMESTAMP:{}\n#Add your message for this simulation here.\n# Don't use '{}' simbol(s). They will be remove from comment\n".format(
					hashline,timestamp,separator) )
			self.simrec['message-timestamp'] = os.stat(self.simrec['messagefile']).st_mtime
			os.system( getconfig("/editor","\'nano\'") + " " + self.simrec['messagefile'] + " &")
	def simdbwrite(self, filename):
		if self.simrec is None:
			self.logger.error("----------------------------------------------------")
			self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.generate)")
			self.logger.error("		     : Cannot write record which wasn't recorded")
			self.logger.error("----------------------------------------------------")		
			if self.pure                 : raise ValueError("Cannot write record which wasn't recorded")
			return False
		if self.simrec["message"] is None:
			nstp = os.stat(self.simrec['messagefile']).st_mtime
			if nstp == self.simrec['message-timestamp']:
				self.logger.error("----------------------------------------------------")
				self.logger.error("SIMMETHODS: METHODS ERROR(simtools.simdb.simmethods.generate)")
				self.logger.error("		     : Cannot record simdb. No message was provided")
				self.logger.error("----------------------------------------------------")		
				if self.pure                 : raise ValueError("Cannot record simdb. No message was provided")
				return False
			fsize = os.path.getsize(self.simrec['messagefile'])
			with open(self.simrec['messagefile']) as fd:
				self.simrec["message"] = fd.read(fsize)
			os.remove(self.simrec['messagefile'])
		self.simrec["message"] = re.sub(r"\n", r"\\n",\
							re.sub(r"\r", r"\\r",\
								re.sub(r"#.*\n",r"",\
									re.sub(self.simrec["separator"], ".",self.simrec["message"]) 
								) 
							) 
						).strip(' \t\n\r')

		result = simdb.write2db(filename, 
			self.simrec["rectype"], self.simrec["record"], self.simrec["message"],
			timestamp=self.simrec["timestamp"], hashline = self.simrec["hash"],
			separator = self.simrec["separator"])
		self.simrec = None
		return result
		
	def genconfile(self,xfile, hsymbol = "/", separator=";", unresolved = False, close=True):
		record,_ = self.gendbrecord(hsymbol = hsymbol, separator=separator, unresolved =  unresolved)
		now = datetime.now()
		if type(xfile) is str:
			xfile =  open(xfile, "w")
			close = True
		if not type(xfile) is file:ValueError("genconfile: xfile should be string or file object {} is given.".format(type(xfile)))
			
		xfile.write("#--------------------------------------------------------------------------------#\n")
		xfile.write("#--- This CONFIG File Automatically Generated by SIMMETHOD module of SIMTOOLS ---#\n")
		xfile.write("#---                   DATE and TIME: %04d-%02d-%02d, %02d:%02d:%02d                   ---#\n"%(now.year, now.month, now.day, now.hour, now.minute, now.second))
		xfile.write("#--------------------------------------------------------------------------------#\n")
		for v in record.split(';'):
			xfile.write(v+"\n")
		xfile.write("#--------------------------------------------------------------------------------#\n")
		if close: xfile.close()
						
	def printmethods(self,tree=None,space=""):
		if tree is None: tree = self.methods
		prn = ""
		for prim, name in tree.printnames():
			rep = str(prim)
			if name is None:
				prn += rep
				continue
			if len(rep) < 31:
				for x in xrange(31-len(rep)):rep += " "
				
			if type(tree[name]) is str:
				rep += " : \"{}\"\n".format(tree[name])
			elif self.is_lambda(tree[name]):
				if name in self.namespace:
					rep += " : {}\n".format(self.namespace[name])
				else:
					rep += " : \"{}\"\n".format(tree[name])
			else:
				rep += " : {}\n".format(tree[name])	
			if len(rep)>544:
				rep = rep[:31+127]+ " ~ ... ~ " + rep[-127:]
			prn += rep
		return prn
	def logmethods(self, tree=None, space=""):
		if tree is None: tree = self.methods
		for prim, name in tree.printnames():
			rep = str(space+prim)
			if name is None:
				self.logger.info(rep[:-1])
				continue
			if len(rep) < 31:
				for x in xrange(31-len(rep)):rep += " "
				
			if type(tree[name]) is str:
				rep = rep + " : \"{}\"".format(tree[name])
			elif self.is_lambda(tree[name]):
				if name in self.namespace:
					rep = rep + " : {}".format(self.namespace[name])
				else:
					rep = rep + " : \"{}\"".format(tree[name])
			else:
				rep = rep + " : {}".format(tree[name])	
			self.logger.info(rep)
		
	
if __name__ == "__main__":
	#CHECK LOGGER
	logging.basicConfig(format='%(asctime)s:%(name)-33s%(lineno)-6d%(levelname)-8s:%(message)s', level=logging.DEBUG)

	#SET DEFAULT PARAMETERS
	p = simmethods(
		default =[
			("/E-population/n = 80","Number of neurons in excitatory population"),
			("/I-population/n = 20","Number of neurons in inhibitory population"),
			("/connections/pee = 0.02","Probability of connection inside E population"),
			("/connections/pii = 0.2","Probability of connection inside I population"),
			("/iterconnection=float(@/E-population/n@)**2*@/connections/pee@+float(@/I-population/n@)**2*@/connections/pii@", "Total number of connections"),
			#AN ERROR>>("/Xiterconnection=float(@/XE-population/n@)**2*@/connections/pee@+float(@/I-population/n@)**2*@/connections/pii@", "Total number of connections")
		],
		argvs = sys.argv[1:],
		target = 'p',
		localcontext = globals()
		)
	
	# GENERATE
	p.generate()
	print p.printmethods()
	p.logmethods()
	if "/nested_function/functionB" in p:
		print p["/nested_function/functionB"](5)
	if "/nested_function" in p:
		print "TEST NESTED FUNCTIONS:"
		print p["/nested_function/functionA"](2,3)
		print p["/nested_function/functionB"](2)
		print
	print "-----------------------------------------------------"
	print "---                   Reset tree                  ---"
	p.methods = methodtree()
	print "GENERATE only connections:"
	p.generate(only="/connections/*", localcontext=globals())
	print p.printmethods()
	print
	print "GENERATE everything else except connections because they had been generated before and iterconnection" 
	p.generate(prohibit=["/connections/*","/iterconnection"], localcontext=globals())
	print p.printmethods()
	print 
	print "GENERATE only iterconnection" 
	p.generate(only="/iterconnection", localcontext=globals())
	print p.printmethods()
	print "CHECK procedure"
	print " /connections/pee : ","/connections/pee" in p
	print " /connections     : ","/connections" in p
	print " /connections/pxx : ","/connections/pxx" in p
	print "WRITE test.db"
	p.simdbrecord(message="Test record")
	p.simdbwrite("test.db")
	
