import os, hashlib, re, sys, time, logging, zlib, simmethods
from random import randint
from simconfig import getconfig
from datetime import datetime
#from git import Repo

str2timestamp = lambda formatstr: tuple( [ int(re.sub(",0",",",x) if len(x) < 3 else re.sub(",0",",",x))\
														for x in formatstr.lstrip("(").rstrip(")").split(",") ] )
timestamp2str = lambda timestamp: "%04d/%02d/%02d %02d:%02d:%02d #%03d"%timestamp
timestamp2timestamp = lambda timestamp: "(%04d,%02d,%02d,%02d,%02d,%02d,%03d)"%timestamp

def write2db(simdbfile,rectype,record, message, timestamp=None, hashline=None, separator=";", pure=False):
	"""
	@function write2db writes new record to simdb file
	@param   simdbfile - a name of simdb file. 
	                     If file doesn't exist it will be create
	@param   rectype   - string of record type
	@oaran   record    - fully formated record
	@param   message   - string of message of this record
	@option  timestamp - string of tome stamp
	                   - if None, it will be generated at the moment of evoke
	@option  hashline  - a string of hash sum
	                   - if None, it will be calculated over record parameter
	@option  separator - separator for current record
	@option  pure      - If true and file unreachable,
	                     function raises an exception
	@return  True      - if record is successfully written
	         False     - otherwise
	"""
	if timestamp is None:
		now = datetime.now()
		timestamp = (now.year, now.month, now.day, now.hour, now.minute, now.second, randint(0,999))
		timestamp = timestamp2timestamp(timestamp)
	elif type(timestamp) is list or type(timestamp) is tuple:
		if len(timestamp) != 7:
			logger = logging.getLogger("simtools.simdb.write2db")
			logger.error("Time Stamp must have 7 fields, given :{}".format(len(timestamp)))
			if pure: raise ValueError("Time Stamp must have 7 fields, given :{}".format(len(timestamp)))
			return False
		elif any([ not type(x) is int for x in timestamp ]):
			logger = logging.getLogger("simtools.simdb.write2db")
			logger.error("Not all field of time stamp are integers:{}".format([ (x,type(x)) for x in timestamp ]))
			if pure: raise ValueError("Not all field of time stamp are integers:{}".format([ (x,type(x)) for x in timestamp ]))
			return False
		timestamp = timestamp2timestamp(timestamp)
	elif not type(timestamp) is str:
			logger = logging.getLogger("simtools.simdb.write2db")
			logger.error("Time Stamp must be a list /tuple with 7 integers or string. {} given".format(type(timestamp)))
			if pure: raise ValueError("Time Stamp must be a list /tuple with 7 integers or string. {} given".format(type(timestamp)))
			return False
	if hashline is None:
		hashline=hashlib.sha1(record).hexdigest()
	recheadersize = len(rectype+separator+timestamp+separator+hashline)
	record = rectype+separator+timestamp+separator+hashline+separator+message+separator+record
	try:
		fsize = os.path.getsize(simdbfile)
	except:
		fsize = 0
		try:
			with open(simdbfile,"w") as fd: pass
		except BaseException as e:
			logger = logging.getLogger("simtools.simdb.write2db")
			logger.error("File \'{}\' cannot be written: {}".format(simdbfile,e))
			if pure: raise ValueError("File \'{}\' cannot be written: {}".format(simdbfile,e))
			return False
	with open(simdbfile,"r+") as fd:
		if fsize < 32:
			fd.write(record + "\n")
			if recheadersize != 73:
				record = "{},".format({'records':[0],'version':3,'default-separator':separator,'headersize':{0:recheadersize} })
			else:
				record = "{},".format({'records':[0],'version':3,'default-separator':separator })
			fd.write(record+"% 32d"%len(record) )
		else:	
			fd.seek(-32,2)
			idx = int( fd.read(32).lstrip("0") )
			fd.seek(-idx-32,2)
			footer = fd.read(idx-1)
			footer = eval( footer )
			if type(footer) is list:
				footer={'records':footer,'version':3,'default-separator':separator}
			if not 'default-separator' in footer:
				if not 'separators' in footer: footer['separators'] = {}
				footer['separators'][len(footer['records'])]=separator
			else:
				if footer['default-separator'] != separator :
					if not 'separators' in footer: footer['separators'] = {}
					footer['separators'][len(footer['records'])]=separator
			if recheadersize != 73:
				if not 'headersize' in footer: footer['headersize'] = {}
				footer['headersize'][len(footer['records'])]=recheadersize
			fd.seek(-idx-32,2)
			fd.write(record + "\n")
			footer['records'].append(fsize-idx-32)
			record = "{},".format(footer)
			fd.write(record+"% 32d"%len(record) )
		return True
#DB>>
#print "create ABC record"
#write2db(sys.argv[1],"SIMREC","ABC","ABC record")
#print "create KLMN record"
#write2db(sys.argv[1],"SIMREC","KLMN","KLMN record",separator=":+:")
#print "create XYZ record"
#write2db(sys.argv[1],"SIMREC","XYZ","XYZ record",
#hashline="---------------------------------------------------------------------------"
#)
#exit(0)
##<<DB
def simdbV01toV03(infile, outfile):
	if not os.access(infile,os.R_OK):
		print "SIMDB: V01 to V03 convertor: Cannot access file : {}".format(infile)
		exit(1)
	with open(infile) as fdin:
		for ln,line in enumerate( fdin.readlines() ):
			field = line[:-1].split(":")
			if field[0][:5] != "SIMDB": continue
			TIMESTAMP = str2timestamp(field[0][16:])
			TIMESTAMP = tuple(list(TIMESTAMP)+[0] )
			p = simmethods.simmethods(target='p')
			p.readargvs(field[1:])
			p.generate(localcontext=locals(),text=True)
			p.simdbrecord(message="-Rec #{} in {}".format(ln,infile), timestamp = TIMESTAMP,zipped=True )
			p.simdbwrite(outfile)
			del p, r

def simdbV02toV03(infile, outfile):
	if not os.access(infile,os.R_OK):
		print "SIMDB: V02 to V03 convertor: Cannot access file : {}".format(infile)
		exit(1)
	db = simdb(infile)
	for recn in xrange(db.reclist):
		rtype,rtime,rhash,rmes,m = db.getrecord(recn)
		if rtype == 'SIMREC' or rtype == 'SIMZIP':
			m.generate(target='m', localcontext=locals())
			m.simdbrecord(message=rmes, timestamp = rtime)
			m.simdbwrite(outfile)
			del m
#		elif rtype == "GITREC":
#			if not write2db(rtype,m,rmes,timestamp=rtime, hashline=rhash):
#				print "Cannor write GIT record : {}".format(rtype,rtime,rhash,rmes,m)
	
class gitrec:
	def __init__(self, names=None, separator=";", directory ="."):pass
#		now = datetime.now()
#		self.timestamp = (now.year, now.month, now.day, now.hour, now.minute, now.second, randint(0,999))
#		if names is None: names = "-a"
#		if type(names) is str: names = [ names ]
#		assert type(names) is list
#		self.repo = Repo(directory)
#		self._hash = self.repo.head.commit.hexsha
#		self.separator = separator
#		os.system(getconfig("/git","\'git\'")+" commit"+reduce(lambda x,y:x+" "+y,names," ")+" &")
	def writerecord(self, filename): pass
#		if self._hash == self.repo.head.commit.hexsha: return False
#		return write2db(filename,
#			"GITREC",'',
#			self.repo.head.commit.message.strip(" \t\r\n").replace("\n","\\n"),
#			timestamp=self.timestamp, hashline=self.repo.head.commit.hexsha,
#			separator = self.separator)	

class simdb:
	def __init__(self,filename,pure=False):
		self.filename = filename
		self.pure = pure
		self.logger = logging.getLogger("simtools.simdb.simdb")
		if not os.access(filename,os.R_OK):
			logging.error("----------------------------------------------------")
			logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.__init__)")
			logging.error("		     : Cannot access file : {}".format(filename))
			logging.error("----------------------------------------------------")		
			if self.pure             : raise ValueError("Cannot access file : {}".format(filename))

		self.update()
	#def error(self, ErrMSG, function):
		#if self.pure                 : raise ValueError(ErrMSG)
		#elif  not self.logger is None: self.logger.error(ErrMSG)
		#else                         : sys.stderr.write("SIMDB: READER ERROR(simtools.simdb.simdb{})\n\t{}\n---------------\n\n".format(function, ErrMSG) )
	def update(self):
		try:
			self.fsize = os.path.getsize(self.filename)
		except:
			self.fsize = 0
			with open(self.filename,"w") as fd: pass
		if self.fsize < 32:
			self.reclist = []
			self.rechash = []
			self.rectstp = []
			self.rectype = []
			self.version = 0
			self.separators = {}
			self.defaul_separator = "/"
		else:
			self.rechash = []
			self.rectstp = []
			self.rectype = []
			self.separators = {}
			self.headersize = {}
			self.defaul_separator = ";"
			with open(self.filename,"r+") as fd:
				fd.seek(-32,2)
				idx = int( fd.read(32).lstrip("0") )
				fd.seek(-idx-32,2)
				footer = fd.read(idx-1)
				footer= eval(footer)
				if type(footer) is list:
					self.reclist = footer
					self.version = 2
					self.defaul_separator = ";"
					self.headersize = {}
				else:
					if not 'records' in footer:
						logging.error("----------------------------------------------------")
						logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.update)")
						logging.error("		     : Cannot find records in data base.... ERROR: %s"%(self.filename))
						logging.error("----------------------------------------------------")		
						if self.pure             : raise ValueError("Cannot find records in data base.... ERROR: %s"%(self.filename))
					elif not type(footer['records']) is list:
						logging.error("----------------------------------------------------")
						logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.update)")
						logging.error("		     : Wrong type {} of records header in data base {}. Must be a list".format(type(footer['records']),self.filename))
						logging.error("----------------------------------------------------")		
						if self.pure             : raise ValueError("Wrong type {} of record in data base {}. Must be a list".format(type(footer['records']),self.filename))
					self.reclist = footer['records']
					if not 'version' in footer:
						logging.error("----------------------------------------------------")
						logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.update)")
						logging.error("		     : Cannot find version in data base.... ERROR: %s"%(self.filename))
						logging.error("----------------------------------------------------")		
						if self.pure             : raise ValueError("Cannot find records in data base.... ERROR: %s"%(self.filename))
					elif not type(footer['version']) is int:
						logging.error("----------------------------------------------------")
						logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.update)")
						logging.error("		     : Wrong type {} of version in data base {}. Must be a list".format(type(footer['version']),self.filename))
						logging.error("----------------------------------------------------")		
						if self.pure             : raise ValueError("Wrong type {} of version in data base {}. Must be a list".format(type(footer['version']),self.filename))
					self.version = footer['version']
					if 'separators' in footer:
						self.separators = footer['separators']
					if 'headersize' in footer:
						self.headersize = footer['headersize']
					if 'default-separator' in footer:
						self.defaul_separator = footer['default-separator']
					
				for recn,recp in enumerate(self.reclist):
					fd.seek(recp)
					if recn in self.headersize:
						rec = fd.read(self.headersize[recn])
					else:
						rec = fd.read(73)
					if recn in self.separators:
						rec = rec.split(self.separators[recn])
					else:
						rec = rec.split(self.defaul_separator)
					if rec[0] != "SIMREC" and  rec[0] != "SIMZIP" and  rec[0] != "GITREC": continue
					try:
						timestamp = str2timestamp( rec[1] )
					except BaseException as e:
						logging.error("----------------------------------------------------")
						logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.update)")
						logging.error("		     : Cannot read timestamp for record %d: %s.... ERROR: %s"%(idx,rec[1],e))
						logging.error("----------------------------------------------------")		
						if self.pure             : raise ValueError("Cannot read timestamp for record %d: %s.... ERROR: %s"%(idx,rec[1],e))
						continue
					try:
						phash = eval("\""+rec[2]+"\"")
					except BaseException as e:
						logging.error("----------------------------------------------------")
						logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.update)")
						logging.error("		     : Cannot read hash for record %d: %s.... ERROR: %s"%(idx,rec[2],e))
						logging.error("----------------------------------------------------")		
						if self.pure             : raise ValueError("Cannot read hash for record %d: %s.... ERROR: %s"%(idx,rec[2],e))
						continue
					self.rechash.append( phash )
					self.rectstp.append( timestamp )
					self.rectype.append( rec[0] )
	def getcomments(self):
		comments = []
		with open(self.filename) as fd:
			for recn,recp in enumerate(self.reclist):
				fd.seek(recp)
				record = fd.readline()		
				if recn in self.separators:
					record = record.split(self.separators[recn])
				else:
					record = record.split(self.defaul_separator)
				if len(record)<3:
					logging.error("----------------------------------------------------")
					logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.getcomments)")
					logging.error("		     : Cannot read comment for record # {}".format(recn))
					logging.error("----------------------------------------------------")		
					if self.pure             : raise ValueError("Cannot read comment for record # {}".format(recn))
					comments.append( "ERROR" )
				elif record[0] == "GITREC":pass
#					if len(record) < 4:
#						repo = Repo(".")
#						for com in  list(repo.iter_commits()):
#							if com.hexsha == record[2].strip(" \t\r\n"):
#								comments.append(com.message.strip(" \t\r\n"))
#								break
#					else:
#						comments.append(record[3])
				elif len(record)<4:
					logging.error("----------------------------------------------------")
					logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.getcomments)")
					logging.error("		     : Cannot read comment for record #. {}".format(recn))
					logging.error("----------------------------------------------------")		
					if self.pure             : raise ValueError("Cannot read comment for record #. {}".format(recn))
					comments.append( "ERROR" )
				else:
					comments.append(record[3])
		return comments
	def readrec(self, indentifier, lst=False):
		if   type(indentifier) is int: return self.readrecint(indentifier, lst=lst)
		elif type(indentifier) is str: return self.readrechash(indentifier, lst=lst)
		elif type(indentifier) is tuple: 
			if len(indentifier) != 7:
				logging.error("----------------------------------------------------")
				logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrec)")
				logging.error("		     : Cannot time stamp {} does not have 7 fields.... "%(indentifier))
				logging.error("----------------------------------------------------")		
				if self.pure             : raise ValueError("Cannot time stamp {} does not have 7 fields.... "%(indentifier))
				return (),"","",None
			return self.readrectimestamp(indentifier, lst=lst)
		logging.error("----------------------------------------------------")
		logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrec)")
		logging.error("		     : Indentifier should be int or string or tuple with 7 elements, given {}: {}.... ".format(type(indentifier),indentifier))
		logging.error("----------------------------------------------------")		
		if self.pure             : raise ValueError("Indentifier should be int or string or tuple with 7 elements, given {}: {}.... ".format(type(indentifier),indentifier))
		return "ERROR",(),"","",None
	def readrectimestamp(self, timestamp, lst=False):
		if not timestamp in self.rectstp:
			logging.error("----------------------------------------------------")
			logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrectimestamp)")
			logging.error("		     : Cannot find time stamp %s in file.... "%str(timestamp))
			logging.error("----------------------------------------------------")		
			if self.pure             : raise ValueError("Cannot find time stamp %s in file.... "%str(timestamp))
			return "ERROR",(),"","",None
		idx = self.rectstp.index(timestamp)
		return self.readrecint(idx, lst=lst)
	def readrechash(self, phash, lst=False):
		if not phash in self.rechash:
			logging.error("----------------------------------------------------")
			logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrechash)")
			logging.error("		     : Cannot find hash %s in file.... "%str(phash))
			logging.error("----------------------------------------------------")		
			if self.pure             : raise ValueError("Cannot find hash %s in file.... "%str(phash))
			return "ERROR",(),"","",None
		idx = self.rechash.index(phash)
		return self.readrecint(idx, lst=lst)
	def readrecint(self, idx, lst=False):
		if idx >= len(self.reclist) or idx < -len(self.reclist):
			logging.error("----------------------------------------------------")
			logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrecint)")
			logging.error("		     : Index idx=%d is out of range %d : %d in current sindb %s"%(idx,-len(self.reclist),len(self.reclist),self.filename))
			logging.error("----------------------------------------------------")		
			if self.pure             : raise ValueError("Index idx=%d is out of range %d : %d in current sindb %s"%(idx,-len(self.reclist),len(self.reclist),self.filename))
			return "ERROR",(),"","",None
		with open(self.filename) as fd:
			fd.seek(self.reclist[idx])
			record = fd.readline()
		if idx in self.separators:
			argv = record.split(self.separators[recn])
		else:
			argv = record.split(self.defaul_separator)
		
		if argv[0] != "SIMREC" and argv[0] != "SIMZIP" and argv[0] != "GITREC":
			logging.error("----------------------------------------------------")
			logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrecint)")
			logging.error("		     : Record %d does not have a header. Read: %s...."%(idx,record[:20]))
			logging.error("----------------------------------------------------")		
			if self.pure             : raise ValueError("Record %d does not have a header. Read: %s...."%(idx,record[:20]))
			return "ERROR",(),"","",None
		try:
			timestamp = str2timestamp( argv[1] )
		except BaseException as e:
			logging.error("----------------------------------------------------")
			logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrecint)")
			logging.error("		     : Cannot read timestamp for record %d: %s.... ERROR: %s"%(idx,argv[1],e))
			logging.error("----------------------------------------------------")		
			if self.pure             : raise ValueError("Cannot read timestamp for record %d: %s.... ERROR: %s"%(idx,argv[1],e))
			timestamp = None
		try:
			phash = eval("\""+argv[2].strip(" \t\r\n")+"\"")
		except BaseException as e:
			logging.error("----------------------------------------------------")
			logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrecint)")
			logging.error("		     : Cannot read hash for record %d: %s.... ERROR: %s"%(idx,argv[2],e))
			logging.error("----------------------------------------------------")		
			if self.pure             : raise ValueError("Cannot read hash for record %d: %s.... ERROR: %s"%(idx,argv[2],e))
		if argv[0] == "SIMREC" or argv[0] == "SIMZIP":
			try:
				comment = eval("\""+argv[3]+"\"")
			except BaseException as e:
				logging.error("----------------------------------------------------")
				logging.error("SIMMETHODS: DATABASE ERROR(simtools.simdb.simdb.readrecint)")
				logging.error("		     : Cannot read comment for record %d: %s.... ERROR: %s"%(idx,argv[3],e))
				logging.error("----------------------------------------------------")		
				if self.pure             : raise ValueError("Cannot read comment for record %d: %s.... ERROR: %s"%(idx,argv[3],e))
#		else:
#			if len(argv) <4:
#				repo = Repo(".")
#				for com in  list(repo.iter_commits()):
#					if com.hexsha == phash:
#						comment = "GIT:"+com.message.strip(" \t\r\n")
#						break
#			else:
#				comment = "GIT:" +  argv[3]
		if argv[0] == "GITREC":
			return None
#			return argv[0],timestamp, phash, comment, []
		elif argv[0] == "SIMZIP":
			decom = zlib.decompress(argv[4].decode("base64"))
			if idx in self.separators:
				argv = decom.split(self.separators[recn])
			else:
				argv = decom.split(self.defaul_separator)
			if not lst: argv = simmethods.simmethods( presets=argv )
			return "SIMREC",timestamp, phash, comment, argv
		else: #argv[0] == "SIMREC"
			if lst:
				return argv[0],timestamp, phash, comment, argv[4:]
			return argv[0],timestamp, phash, comment,simmethods.simmethods( presets=argv[4:] )
			
	

if __name__ == "__main__":
	import sys
	print "\n ------------------------- \n"
	p = simmethods.simmethods(target="p",localcontext=globals())
	p.generate()
	p['/a'] = 1
	p['/b'] = 2
	p['/c'] = 3
	simcorder1 = simrec(tree=p,comment="simple record")
	print "SIMPLE", simcorder1.getrecord()
	print "\n ------------------------- \n"

	print "BUILD methods"
	p = simmethods.simmethods(
		delault = [
			("/neuron/c=1",   "Cap"),
			("/neuron/r=1.1", "Res"),
			("/Population-E/neuron = @/neuron@", "population of E neuron"),
			("/Population-E/n      = 20", "number of neurons in population"),
			("/Population-E/E-synaps/tau = 0.1", "Tau of excitatory synapse in excitatory population"),
			("/Population-I/neuron = @/neuron@", "population of I neuron"),
			("/Population-I/n      = 10", "number of neurons in population"),
			("/Population-I/E-synaps/tau = 0.1", "Tau of excitatory synapse in inhibitory population"),
			("/Total = @/Population-E/n@ * @/Population-I/n@","Total number of neurons"),
			("/Edist = lambda x,sigma: exp(-x**2/sigma**2)","Distribution of propability"),
			("/LOG = \'myprogram.log\'","Log file name"),
		],
		target="p",localcontext=globals()
	)
	p.readargvs(sys.argv[1:])
	p.generate()
	recorder2 = simrec(tree=p, comment="Tree based methods")
	print "COMPLICATED", recorder2.getrecord()
	print "\n ------------------------- \n"
	exit(0)
	recorder1.writerecord("test.simdb")
	recorder2.writerecord("test.simdb")
	
	#SIMDB TEST
	sdb = simdb("test.simdb")
	print "\n ------------------------- \n"
	print   " ---    SIMDB TESTS    --- "
	print   " ---                   --- "
	print   " ---   ADDED RECORDS   --- "

	typ,tsp,hs,cm,d=sdb.readrec(-1)
	print "LAST RECORD"
	print "TYPE      : ",typ
	print "TIMESTAMP : ",tsp
	print "HASH      : ",hs
	print "COMMENT   : ",cm
	d.generate(target='d',localcontext=globals())
	print d.printmethods()
	print

	typ,tsp,hs,cm,d=sdb.readrec(-2)
	print "RECORD BEFORE LAST"
	print "TYPE      : ",typ
	print "TIMESTAMP : ",tsp
	print "HASH      : ",hs
	print "COMMENT   : ",cm
	d.generate(target='d',localcontext=globals())
	print d.printmethods()
	print

	typ,tsp,hs,cm,d=sdb.readrec(4)
	print "\n ------------------------- \n"
	print "RECORD #4"
	print "TYPE      : ",typ
	print "TIMESTAMP : ",tsp
	print "HASH      : ",hs
	print "COMMENT   : ",cm
	d.generate(target='d',localcontext=globals())
	print d.printmethods()
	print
	
	typ,tsp,hs,cm,d=sdb.readrec("ff95eae5711e158f3082d2350f2f4cbcc3f82b6c")
	print "RECORD by HASH"
	print "TYPE      : ",typ
	print "TIMESTAMP : ",tsp
	print "HASH      : ",hs
	print "COMMENT   : ",cm
	d.generate(target='d',localcontext=globals())
	print d.printmethods()
	print

	typ,tsp,hs,cm,d=sdb.readrec((2017, 2, 22, 0, 29, 0, 896))
	print "RECORD by TIMESTAMP"
	print "TYPE      : ",typ
	print "TIMESTAMP : ",tsp
	print "HASH      : ",hs
	print "COMMENT   : ",cm
	if not d is None:
		d.generate(target='d',localcontext=globals())
		print d.printmethods()
	print
	
	typ,tsp,hs,cm,d=sdb.readrec(((2017,03,05,23,54,16,575)))
	print "RECORD by TIMESTAMP"
	print "TYPE      : ",typ
	print "TIMESTAMP : ",tsp
	print "HASH      : ",hs
	print "COMMENT   : ",cm
	if not d is None:
		d.generate(target='d',localcontext=globals())
		print d.printmethods()
	print
	
	p.generate(target='p',localcontext=globals())
	recorder4 = simrec(tree=p)
	import time
	time.sleep(30)
	while not recorder4.writerecord("test.simdb"):
		time.sleep(30)
		
	print "\n ------------------------- \n"
#	print "GIT RECORD"
#	gr = gitrec(names=sys.argv[0])
#	time.sleep(40)
#	gr.writerecord("test.simdb")
#	print "\n ------------------------- \n"
	
	print "TWO RECORDS WITH DEEP DIFF"
	p.generate(target='p',localcontext=globals())
	recorder2 = simrec(tree=p, comment="default parameters")
	recorder2.writerecord("test.simdb")
	p["/Population-I/E-synaps/tau"] = 100.
	recorder2 = simrec(tree=p, comment="updated parameters")
	recorder2.writerecord("test.simdb")
	
	
