# Credit for most of this code is due to Alex Drlica-Wagner
import numpy as np
import fitsio
import glob

def go():

    '''
    Note: to read a subset of data, columns = ['column1','column2']; data = loadfiles(filenames,columns=columns)
    '''

    print
    option = raw_input('Enter clusters or members: ')

    if (option == 'cluster' or option == 'clusters'):

        # Define locations of input/output catalogs
        filenames = sorted(glob.glob('/data/des40.a/data/jburgad/clusters/outputs_brian/feb15/lgt5_mof_bin01_clusters_*'))
        data = loadfiles(filenames) # Call function and load all data sets
        out_dir = '/data/des40.a/data/jburgad/clusters/outputs_brian/feb15/'
        outfile = out_dir + 'test.fit'

        # Define afterburner column names for printing
        hostid = data['MEM_MATCH_ID'][:]
        ra = data['RA'][:]
        dec = data['DEC'][:]
        z = data['Z'][:]
        r2 = data['R200'][:]
        m2 = data['M200'][:]
        n2 = data['N200'][:]
        lamb = data['LAMBDA_CHISQ'][:]
        grs = data['GR_SLOPE'][:]
        gri = data['GR_INTERCEPT'][:]
        grmr = data['GRMU_R'][:]
        grmb = data['GRMU_B'][:]
        grsr = data['GRSIGMA_R'][:]
        grsb = data['GRSIGMA_B'][:]
        grwr = data['GRW_R'][:]
        grwb = data['GRW_B'][:]
        ris = data['RI_SLOPE'][:]
        rii = data['RI_INTERCEPT'][:]
        rimr = data['RIMU_R'][:]
        rimb = data['RIMU_B'][:]
        risr = data['RISIGMA_R'][:]
        risb = data['RISIGMA_B'][:]
        riwr = data['RIW_R'][:]
        riwb = data['RIW_B'][:]
        izs = data['IZ_SLOPE'][:]
        izi = data['IZ_INTERCEPT'][:]
        izmr = data['IZMU_R'][:]
        izmb = data['IZMU_B'][:]
        izsr = data['IZSIGMA_R'][:]
        izsb = data['IZSIGMA_B'][:]
        izwr = data['IZW_R'][:]
        izwb = data['IZW_B'][:]
        grmbg = data['GRMU_BG'][:]
        grsbg = data['GRSIGMA_BG'][:]
        grwbg = data['GRW_BG'][:]
        flags = data['FLAGS'][:]
        zerr = data['Z_ERR'][:]

        # Now print out the combined catalog
        fits = fitsio.FITS(outfile,'rw', clobber=True)
        names = ['MEM_MATCH_ID','RA','DEC','Z','R200','M200','N200','LAMBDA_CHISQ','GR_SLOPE','GR_INTERCEPT','GRMU_R','GRMU_B','GRSIGMA_R','GRSIGMA_B','GRW_R','GRW_B','RI_SLOPE','RI_INTERCEPT','RIMU_R','RIMU_B','RISIGMA_R','RISIGMA_B','RIW_R','RIW_B','IZ_SLOPE','IZ_INTERCEPT','IZMU_R','IZMU_B','IZSIGMA_R','IZSIGMA_B','IZW_R','IZW_B','GRMU_BG','GRSIGMA_BG','GRW_BG','FLAGS','Z_ERR']
        array_list = [hostid,ra,dec,z,r2,m2,n2,lamb,grs,gri,grmr,grmb,grsr,grsb,grwr,grwb,ris,rii,rimr,rimb,risr,risb,riwr,riwb,izs,izi,izmr,izmb,izsr,izsb,izwr,izwb,grmbg,grsbg,grwbg,flags,zerr]
        fits.write(array_list, names=names)

    elif (option == 'member' or option == 'members'):

        filenames = sorted(glob.glob('/data/des40.a/data/jburgad/clusters/outputs_brian/feb15/lgt5_mof_bin01_members_*'))
        data = loadfiles(filenames) # Call function and load all data sets
        out_dir = '/data/des40.a/data/jburgad/clusters/outputs_brian/feb15/'
        outfile = out_dir + 'test.fit'

        # Define afterburner column names for printing (disregard variable names)
        galid = data['COADD_OBJECTS_ID'][:]
        hid = data['HOST_HALOID'][:]
        raa = data['RA'][:]
        decc = data['DEC'][:]
        zp = data['ZP'][:]
        zpe = data['ZPE'][:]
        mg = data['MAG_AUTO_G'][:]
        mr = data['MAG_AUTO_R'][:]
        mi = data['MAG_AUTO_I'][:]
        mz = data['MAG_AUTO_Z'][:]
        pr = data['P_RADIAL'][:]
        pz = data['P_REDSHIFT'][:]
        grp = data['GR_P_COLOR'][:]
        rip = data['RI_P_COLOR'][:]
        izp = data['IZ_P_COLOR'][:]
        pmem = data['P_MEMBER'][:]
        amag = data['AMAG_R'][:]
        dist = data['DIST_TO_CENTER'][:]
        grpr = data['GRP_RED'][:]
        grpb = data['GRP_BLUE'][:]
        grpbg = data['GRP_BG'][:]
        ripr = data['RIP_RED'][:]
        ripb = data['RIP_BLUE'][:]
        izpr = data['IZP_RED'][:]
        izpb = data['IZP_BLUE'][:]
        gr0 = data['GR0'][:]
        hz = data['HOST_REDSHIFT'][:]
        hzerr = data['HOST_REDSHIFT_ERR'][:]
        mge = data['MAGERR_AUTO_G'][:]
        mre = data['MAGERR_AUTO_R'][:]
        mie = data['MAGERR_AUTO_I'][:]
        mze = data['MAGERR_AUTO_Z'][:]

        # Now print out the combined catalog
        fits = fitsio.FITS(outfile,'rw', clobber=True)
        names = ['COADD_OBJECTS_ID','HOST_HALOID','RA','DEC','ZP','ZPE','MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z','P_RADIAL','P_REDSHIFT','GR_P_COLOR','RI_P_COLOR','IZ_P_COLOR','P_MEMBER','AMAG_R','DIST_TO_CENTER','GRP_RED','GRP_BLUE','GRP_BG','RIP_RED','RIP_BLUE','IZP_RED','IZP_BLUE','GR0','HOST_REDSHIFT','HOST_REDSHIFT_ERR','MAGERR_AUTO_G','MAGERR_AUTO_R','MAGERR_AUTO_I','MAGERR_AUTO_Z']
        array_list = [galid,hid,raa,decc,zp,zpe,mg,mr,mi,mz,pr,pz,grp,rip,izp,pmem,amag,dist,grpr,grpb,grpbg,ripr,ripb,izpr,izpb,gr0,hz,hzerr,mge,mre,mie,mze]
        #names = ['COADD_OBJECTS_ID','HOST_HALOID','RA','DEC','ZP','ZPE','MAG_AUTO_G','MAG_AUTO_R','MAG_AUTO_I','MAG_AUTO_Z','P_RADIAL','P_REDSHIFT','GR_P_COLOR','RI_P_COLOR','IZ_P_COLOR','P_MEMBER','AMAG_R','DIST_TO_CENTER','GRP_RED','GRP_BLUE','GRP_BG','RIP_RED','RIP_BLUE','IZP_RED','IZP_BLUE','HOST_REDSHIFT','HOST_REDSHIFT_ERR','MAGERR_AUTO_G','MAGERR_AUTO_R','MAGERR_AUTO_I','MAGERR_AUTO_Z']
        #array_list = [hostid,zg,galid,gro,gro_err,gio,gio_err,kri,kri_err,kii,kii_err,iobs,distmod,rabs,iabs,mcMass,taMass,mass,mass_err,ssfr,ssfr_std,mass_weight_age,mass_weight_age_err,best_model,best_zmet,best_chisq,grpc,ripc,izpc,grpm,ripm]
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
