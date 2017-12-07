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
    #members_infile = '/data/des40.a/data/jburgad/clusters/outputs_brian/lgt5_mof_members_full.fit'
    members_infile = '/data/des40.a/data/mariaeli/lambda_star/clusters/output_test_membership/lgt5_mof_members_full.fit'
    #smass_outfile = '/data/des40.a/data/jburgad/clusters/outputs_antonella/lgt5_mof/lgt5_mof_stellar_masses_%04d.fit' %(job)
    smass_outfile = '/data/des40.a/data/mariaeli/lambda_star/clusters/output_test_stellarmass/lgt5_mof/lgt5_mof_stellar_masses_%04d.fit' %(job)
    inputDataDict = helperfunctions.read_afterburner(members_infile,a,b)
    smass.calc(inputDataDict, outfile=smass_outfile, indir=indir, lib="miles")
    return

def compute_csmass():
    # For running cluster stellar mass
    # Note: after running compute_smass and before running compute_csmass, it is necesary to use 
    # 'combine_cat.py' to combine all compute_smass output files for the input of compute_csmass.
    indir = '/data/des60.b/data/palmese/lambda_star/fsps_v3.0_modified_Nov16/OUTPUTS/simha_miles_Nov2016/'
    smass_infile = '/data/des40.a/data/mariaeli/lambda_star/clusters/output_test_stellarmass/lgt5_mof/lgt5_mof_stellar_masses_full.fit'
    clusters_outfile = '/data/des40.a/data/mariaeli/lambda_star/clusters/output_test_stellarmass/lgt5_mof/lgt5_mof_cluster_smass_full.fit'
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

def parallel_compute_smass(batch_start=0,max_of_batch=54788,njobs=100,ncores=20):
    print "running..."
    alist = np.linspace(batch_start,max_of_batch,njobs,endpoint=False,dtype=int)
    Parallel(n_jobs=ncores)(delayed(compute_smass)(a,(max_of_batch-batch_start)/njobs) for a in alist)
    return


