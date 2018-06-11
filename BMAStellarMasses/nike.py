import smass
import helperfunctions
import clusterSMass_orig
import sys
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "/home/s1/jburgad/.local/lib/python2.7/site-packages")

# This file is to be ran using afterburner's outputs as inputs for the stellar the mass code
def compute_smass(a,mem_per_job):
    # For running the stellar masses (takes the longest)
    b = a + mem_per_job
    job = int(b/mem_per_job)
    indir = '/data/des60.b/data/palmese/lambda_star/fsps_v3.0_modified_Nov16/OUTPUTS/simha_miles_Nov2016/'
    #indir = '/data/des60.b/data/palmese/lambda_star/fsps_v3.0_modified_Nov16/OUTPUTS/simha_miles_salpeter_Dec17/'
    members_infile = '/data/des40.a/data/jburgad/clusters/outputs_brian/test/test_m_1cluster.fits'#feb15/lgt5_mof_members_full.fit'
    smass_outfile = '/data/des40.a/data/jburgad/clusters/outputs_antonella/test/test_1cluster_stellar_masses_%05d.fits' %(job)#feb15/lgt5_mof_Nov2016_stellar_masses_%04d.fit' %(job)
    inputDataDict = helperfunctions.read_afterburner(members_infile,a,b)
    smass.calc(inputDataDict, outfile=smass_outfile, indir=indir, lib="miles")
    return

def compute_csmass():
    # For running cluster stellar mass
    # Note: after running compute_smass and before running compute_csmass, it is necesary to use 
    # 'combine_cat.py' to combine all compute_smass output files for the input of compute_csmass.
    indir = '/data/des60.b/data/palmese/lambda_star/fsps_v3.0_modified_Nov16/OUTPUTS/simha_miles_Nov2016/'
    #indir = '/data/des60.b/data/palmese/lambda_star/fsps_v3.0_modified_Nov16/OUTPUTS/simha_miles_salpeter_Dec17/'
    smass_infile = '/data/des40.a/data/jburgad/clusters/outputs_antonella/test/test_1cluster_stellar_masses_full.fits'
    clusters_outfile = '/data/des40.a/data/jburgad/clusters/outputs_antonella/test/test_1cluster_cluster_smass_full.fits'
    clusterSMass_orig.haloStellarMass(filename=smass_infile,outfile=clusters_outfile)
    return

''' Parallel Computing Instructions:

    command: $ python -c "import nike; nike.parallel_compute_smass()"

    There are two parameters that need to be entered for each batch submission
    1) batch_start =  the starting member point for this batch out of all the members. the first batch should begin at 0
    2) max_of_batch = the ending member point for this batch out of all members. to avoid duplicates this number should
                      increase by the same amount batch_start increases when doing multiple batches

    Two other parameters that can be adjusted as necessary are
    3) njobs =        the number of jobs the batch submission is split into. 100 is a good number
    4) ncores =       the total number of cores used by the computing machine at any given time. 20 is suggested for the DES cluster

'''

def parallel_compute_smass(batch_start=0,max_of_batch=550,njobs=10,ncores=20): # njobs is normally = 100
    alist = np.linspace(batch_start,max_of_batch,njobs,endpoint=False,dtype=int)
    Parallel(n_jobs=ncores)(delayed(compute_smass)(a,(max_of_batch-batch_start)/njobs) for a in alist)
    return

#def parallel_compute_smass(batch_start=4579756,max_of_batch=9159512,njobs=100,ncores=20):
#    alist = np.linspace(batch_start,max_of_batch,njobs,endpoint=False,dtype=int)
#    Parallel(n_jobs=ncores)(delayed(compute_smass)(a,(max_of_batch-batch_start)/njobs) for a in alist)
#    return

#def parallel_compute_smass(batch_start=9159512,max_of_batch=13739268,njobs=100,ncores=20):
#    alist = np.linspace(batch_start,max_of_batch,njobs,endpoint=False,dtype=int)
#    Parallel(n_jobs=ncores)(delayed(compute_smass)(a,(max_of_batch-batch_start)/njobs) for a in alist)
#    return

#def parallel_compute_smass(batch_start=13739268,max_of_batch=18319024,njobs=100,ncores=20):
#    alist = np.linspace(batch_start,max_of_batch,njobs,endpoint=False,dtype=int)
#    Parallel(n_jobs=ncores)(delayed(compute_smass)(a,(max_of_batch-batch_start)/njobs) for a in alist)
#    return

#def parallel_compute_smass(batch_start=18319024,max_of_batch=22898780,njobs=100,ncores=20):
#    alist = np.linspace(batch_start,max_of_batch,njobs,endpoint=False,dtype=int)
#    Parallel(n_jobs=ncores)(delayed(compute_smass)(a,(max_of_batch-batch_start)/njobs) for a in alist)
#    return
