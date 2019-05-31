import simmethods, os

def getconfig(name,default):
	"""
	@function getconfig reads configuration parameters and returns a value
	@param  name    - name of parameter
	@param  default - default value
	Function reads first /etc/simtools.conf, then .simtoolsrc in home user 
	directory and then .simtoolsrc in current directory therefore all values
	in local .simtoolsrc have highest priority, while in /etc/simtools.conf 
	have lowest priority
	"""
	simdbvars = simmethods.simmethods(presets=[name+"={}".format(default)], target="simdbvars", localcontext=locals())
	#First global
	infile = "/etc/simtools.conf"
	if os.access(infile,os.R_OK):simdbvars.readfile(infile)
	#Second global for user
	infile = os.environ['HOME']+"/.simtoolsrc"
	if os.access(infile,os.R_OK):simdbvars.readfile(infile)
	#Third local in folder
	infile = ".simtoolsrc"
	if os.access(infile,os.R_OK):simdbvars.readfile(infile)
	simdbvars.generate(localcontext=locals())
	return simdbvars[name]		

if __name__ == "__main__":
	print "X"
	print getconfig("/deep",12)
	print getconfig("/editor",'None')
