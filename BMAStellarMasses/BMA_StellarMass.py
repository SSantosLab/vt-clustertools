# new libraries
import ConfigParser
import logging
from time import time
from os import path
# imports from nike.py below
import smass
import helperFunctions
import clusterSMass_orig
import numpy as np
from joblib import Parallel, delayed


def getConfig(section, item, boolean=False,
		userConfigFile="BMA_StellarMass_Config.ini"):

	configFile = ConfigParser.ConfigParser()
	configFile.read(userConfigFile)

	# if config item not found, raise log warning
	if (not configFile.has_option(section, item)):
		msg = '{item} from [{setion}] NOT found in config file: {userConfigFile}!'.format(
			item=item, section=section,
			userConfigFile=userConfigFile)
		if (section != 'Log'):
			logging.warning(msg)
		else:
			print msg
		return ""

	# else save item value (debug)
	msg = '{item}: {value}'.format(
		item=item, value=configFile.get(section, item))
	if (section != 'Log'):
		logging.debug(msg)
	else:
		print msg

	if (not boolean):
		return configFile.get(section, item)

	else:
		return configFile.getboolean(section, item)


def isOperationSet(operation,section="Operations"):
	return getConfig(boolean=True, section=section,
		item=operation)


def createLog():
	logLevel = getConfig("Log","level")
	logFileName = getConfig("Log","logFile")
	myFormat = '[%(asctime)s] [%(levelname)s]\t%(module)s - %(message)s'
	if logLevel == 'DEBUG':
		logging.basicConfig(
			filename=logFileName,
			level=logging.DEBUG,
			format=myFormat)
	else:
		logging.basicConfig(
			filename=logFileName,
			level=logging.INFO,
			format=myFormat)


def extractTarGz(tarFileName, path):
	import tarfile
	tar = tarfile.open(tarFileName, "r:gz")
	tar.extractall(path=inputPath)
	tar.close()


def getInputPath():
	inputPath = getConfig("Paths","inputPath")
	# if a inputPath is not set, go after the .tar.gz file
	if (not inputPath):

		# if tarFile doesn't exist, abort
		tarName = getConfig("Files","tarFile")

		if (not tarName or not path.isfile(tarName) or
				not tarName.endswith("tar.gz")):

			return ""

		# defining inputPath to uncompress file
		inputPath = "./simha_miles_Nov2016/"
		extractTarGz(tarFileName=tarName, path=inputPath)

	return inputPath


def getStellarMassOutPrefix():
	stellarMassOutPrefix = getConfig("Files","stellarMassOutPrefix")

	if not stellarMassOutPrefix:
		logging.critical("Can't continue without stellarMassOutPrefix defined! Exiting.")
		exit()

	return stellarMassOutPrefix


def combineFits():
	from combineCat import combineBMAStellarMassOutput
	stellarMassOutPrefix = getStellarMassOutPrefix()
	combineBMAStellarMassOutput(stellarMassOutPrefix)


def computeStellarMass(batch, memPerJob):
	# For running the stellar masses (takes the longest)
	batchIndex = batch + memPerJob
	job = int(batchIndex / memPerJob)

	logging.debug('Starting computeStellarMass() with batch = {b}; job = {j}.'.format(
		b = batch, j = job))

	stellarMassOutFile = getConfig("Files","stellarMassOutPrefix") + "{:0>5d}.fits".format(job)

	inPath = getInputPath()
	membersInFile = getConfig("Files","membersInputFile")

	if (not inPath or not membersInFile):
		logging.critical("Can't continue without either inputPath or membersInputFile defined! Exiting.")
		exit()

	inputDataDict = helperFunctions.read_afterburner(membersInFile, batch, batchIndex)

	smass.calc(inputDataDict, outfile=stellarMassOutFile, indir=inPath, lib="miles")

	logging.debug('Returning from computeStellarMass() with batch = {b}; job = {j}.'.format(
		b = batch, j = job))


def computeClusterStellarMass():
	stellarMassFile = getConfig("Files","stellarMassOutPrefix") + 'full.fits'	
	clusterOutFile  = getConfig("Files","clusterStellarMassOutFile")

	logging.info('Computing cluster stellar mass.')
	clusterSMass_orig.haloStellarMass(filename = stellarMassFile, outfile = clusterOutFile)


def parallelComputeStellarMass(batchStart=0,
		batchMax=25936, nJobs=100, nCores=20):
		# nJobs is normally = 100
	batchesList = np.linspace(batchStart, batchMax, nJobs, endpoint=False, dtype=int)

	logging.info('Calling parallelism inside parallelComputeStellarMass().')
	Parallel(n_jobs=nCores)(delayed(computeStellarMass)
		(batch, (batchMax - batchStart) / nJobs) 
		for batch in batchesList)

	# generate concatenated fits file
	logging.info('Combining fits.')
	combineFits()


def main():
	# start logging
	createLog()

	logging.info('Starting BMA Stellar Masses program.')

	# get initial time
	total_t0 = time()

	# check and parallel compute stellar mass,
	#	if it is the case
	if (isOperationSet(operation="stellarMass")):
		logging.info('Starting parallel stellar masses operation.')
		section = "Parallel"

		stellarMass_t0 = time()
		# get parallel information
		batchStart = int(getConfig(section, "batchStart"))
		batchMax   = int(getConfig(section, "batchMax"))
		nJobs 	   = int(getConfig(section, "nJobs"))
		nCores 	   = int(getConfig(section, "nCores"))

		# call function to parallel compute
		parallelComputeStellarMass(batchStart=batchStart,
			batchMax=batchMax, nJobs=nJobs, nCores=nCores)

		# save time to compute stellar mass
		stellarMassTime = time() - stellarMass_t0
		stellarMassMsg = "Stellar Mass (parallel) time: {}s".format(stellarMassTime)
		logging.info(stellarMassMsg)

	# check and compute cluster stellar mass,
	#	if it is the case
	if (isOperationSet(operation="clusterStellarMass")):
		logging.info('Starting cluster stellar mass operation.')
		clusterStellarMassTime_t0 = time()
		computeClusterStellarMass()

		# save time to compute cluster stellar mass
		clusterStellarMassTime = time() - clusterStellarMassTime_t0
		clusterStellarMassMsg = "Cluster Stellar Mass time: {}s".format(clusterStellarMassTime)
		logging.info(clusterStellarMassMsg)

	# save total computing time
	totalTime = time() - total_t0
	totalTimeMsg = "Total time: {}s".format(totalTime)
	logging.info(totalTimeMsg)
	logging.info('All done.')


if __name__ == "__main__":
    main()
