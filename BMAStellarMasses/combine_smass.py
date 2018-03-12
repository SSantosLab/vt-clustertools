# Credit for most of this code is due to Alex Drlica-Wagner
import numpy as np
import fitsio
import glob

def go():

    '''
    Note: to read a subset of data, columns = ['column1','column2']; data = loadfiles(filenames,columns=columns)
    '''

    # Define locations of input/output catalogs 
    filenames = sorted(glob.glob('/data/des40.a/data/jburgad/clusters/outputs_antonella/mar1_xray/xmm_stellar_masses_*'))#feb15/lgt5_mof_Nov2016_stellar_masses_*'))
    data = loadfiles(filenames) # Call function and load all data sets
    out_dir = '/data/des40.a/data/jburgad/clusters/outputs_antonella/mar1_xray/'
    outfile = out_dir + 'test.fit'

    # Define BMA stellar mass column names for printing
    hostid = data['MEM_MATCH_ID'][:]
    zg = data['Z'][:]
    galid = data['ID'][:]
    gro = data['gr_o'][:]
    gro_err = data['gr_o_err'][:]
    gio = data['gi_o'][:]
    gio_err = data['gi_o_err'][:]
    kri = data['kri'][:]
    kri_err = data['kri_err'][:]
    kii = data['kii'][:]
    kii_err = data['kii_err'][:]
    iobs = data['iobs'][:]
    distmod = data['distmod'][:]
    rabs = data['rabs'][:]
    iabs = data['iabs'][:]
    mcMass = data['mcMass'][:]
    taMass = data['taMass'][:]
    mass = data['mass'][:]
    mass_err = data['mass_err'][:]
    ssfr = data['ssfr'][:]
    ssfr_std = data['ssfr_std'][:]
    mass_weight_age = data['mass_weight_age'][:]
    mass_weight_age_err = data['mass_weight_age_err'][:]
    best_model = data['best_model'][:]
    best_zmet = data['best_zmet'][:]
    zmet = data['zmet'][:]
    best_chisq = data['best_chisq'][:]
    grpc = data['GR_P_COLOR'][:]
    ripc = data['RI_P_COLOR'][:]
    izpc = data['IZ_P_COLOR'][:]
    grpm = data['P_RADIAL'][:]
    ripm = data['P_REDSHIFT'][:]
    izpm = data['P_MEMBER'][:]
    dist2c = data['DIST_TO_CENTER'][:]
    grpr = data['GRP_RED'][:]
    grpb = data['GRP_BLUE'][:]
    ripr = data['RIP_RED'][:]
    ripb = data['RIP_BLUE'][:]
    izpr = data['IZP_RED'][:]
    izpb = data['IZP_BLUE'][:]

    # Now print out the combined catalog
    fits = fitsio.FITS(outfile,'rw', clobber=True)
    names = ['MEM_MATCH_ID','Z','ID','gr_o','gr_o_err','gi_o','gi_o_err','kri','kri_err','kii','kii_err','iobs','distmod','rabs','iabs','mcMass','taMass','mass','mass_err','ssfr','ssfr_std','mass_weight_age','mass_weight_age_err','best_model','best_zmet','zmet','best_chisq','GR_P_COLOR','RI_P_COLOR','IZ_P_COLOR','P_RADIAL','P_REDSHIFT','P_MEMBER','DIST_TO_CENTER','GRP_RED','GRP_BLUE','RIP_RED','RIP_BLUE','IZP_RED','IZP_BLUE']
    array_list = [hostid,zg,galid,gro,gro_err,gio,gio_err,kri,kri_err,kii,kii_err,iobs,distmod,rabs,iabs,mcMass,taMass,mass,mass_err,ssfr,ssfr_std,mass_weight_age,mass_weight_age_err,best_model,best_zmet,zmet,best_chisq,grpc,ripc,izpc,grpm,ripm,izpm,dist2c,grpr,grpb,ripr,ripb,izpr,izpb]
    fits.write(array_list, names=names)

    print
    print '--> Status: File Complete <%s>' %outfile
    print

    return

def loadfiles(filenames, columns=None):
    '''
    Read a set of filenames with fitsio.read and return a concatenated array
    '''
    out = []
    i=1
    print
    for f in filenames:
        print 'File %s: %s' %(i,f)
        out.append(fitsio.read(f,columns=columns))
        i+=1

    return np.concatenate(out)
