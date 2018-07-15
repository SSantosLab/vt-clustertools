
# coding: utf-8

# In[1]:


# First some imports that we'll use below
from __future__ import print_function
import treecorr
import fitsio
import numpy
import math
import time
import pprint
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
get_ipython().magic(u'matplotlib inline')
from scipy import stats
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Distance
from astropy.cosmology import WMAP5 as cosmo
from astropy import cosmology
import healpy as hp


# In[2]:


def get_z(data):# set redshift ranges for each of the 25 bins so each bin has the same amount of galaxies
    sorted_data = sorted(data, key= lambda redshift:redshift[Z_COL])
    z_min = []
    z_max = []
    for i in range (0,25):
        zmin = sorted_data[i*(len(data)/25-1)][Z_COL]
        zmax = sorted_data[(i+1)*(len(data)/25-1)][Z_COL]
        z_min.append(zmin)
        z_max.append(zmax)
    return (z_min, z_max)


# In[3]:


def get_theta (z_max):#set the theta min and theta max for each redshift to a physical range of 0.3Mpc - 2.5Mpc
    theta_min = []
    theta_max = []
    for red in range (0,25):
        d1 = cosmo.comoving_distance([z_max[red]])
        thetamin = 0.3*180/(numpy.pi * d1.value)
        thetamax = 2.5*180/(numpy.pi * d1.value)
        theta_min.append(thetamin)
        theta_max.append(thetamax)
    return(theta_min, theta_max)


# In[4]:


def ang2pix (dec_rad, ra_rad): # function that transforms ra and dec into theta and phi to create healpy pixels
    theta = (90 - (dec_rad *180/numpy.pi))*numpy.pi/180
    phi = ra_rad
    nside = int(math.pow(2, 10)) # this number was chosen because the pixels are small enough to accurate shape
                                 # the catalog, but big enough not to loose galaxies
    pix = hp.ang2pix(nside, theta, phi)
    return pix


# In[5]:


def rand (ra_rad, dec_rad, data): # creates random catalog in the shape of galaxy catalog using healpy
    ra_min = numpy.min(ra_rad)
    ra_max = numpy.max(ra_rad)
    dec_min = numpy.min(dec_rad)
    dec_max = numpy.max(dec_rad)
    rand_ra = numpy.random.uniform(ra_min, ra_max, len(data)*10) 
    rand_sindec = numpy.random.uniform(numpy.sin(dec_min), numpy.sin(dec_max), len(data)*10) 
    rand_dec = numpy.arcsin(rand_sindec) #convert back to dec  
    pix_gal = ang2pix(dec_rad, ra_rad) # creates pixels in shape of galaxy catalog
    pix_rand = ang2pix(rand_dec, rand_ra) # creates pixels in shape of randoms (a box)
    ix = numpy.in1d(pix_rand, pix_gal) # finds pixels from the randoms that are also in the galaxy sample
    good_pix = numpy.where(ix) # finds where these pixels coincide (a boolean)
    pix_rand = pix_rand[good_pix] # discards pixels that are not shared with galaxy catalog
    rand_ra = rand_ra[good_pix] #filtersthe points in the random catalog that are not in the shared pixels
    rand_dec = rand_dec[good_pix]
    unique_pix = sorted(list(set(pix_gal))) 
    rand_catalog = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='radians', dec_units='radians')
    return (rand_catalog, rand_ra, rand_dec, unique_pix, pix_gal, pix_rand)


# In[6]:


#this function is used for the jackknifing, creates nine unique subsamples of equal area
def limits (ra_rad, dec_rad,rand_ra, rand_dec, unique_pix, pix_gal, pix_rand):   
    num = int(len(unique_pix)/9) # number of pixels each subsample should have
    num_dif = (len(unique_pix) - num*9) # since subsamples of equal amount of pixels might be decimal, get the difference
    ra_jack = []
    dec_jack = []
    rand_ra_jack = []
    rand_dec_jack = []
    for i in range (0,9):
        if (i >= (9- num_dif)): # adds one pixel to subsample if dividing the amount of pixels by 9 is not integer.
            start = finish
            finish = start + num + 1
            ste = unique_pix[start: finish]
        else:
            start = i*num
            finish = start + num
            ste = unique_pix[start:finish]
        gix = numpy.invert(numpy.isin (pix_gal, ste)) # gets the indexes in the galaxy catalog where the pixels are not in the subsample
        rix = numpy.invert(numpy.isin (pix_rand, ste))# gets the indexes in the rand catalog where the pixels are not in the subsample
        gal_jack = (numpy.where(gix))
        rand_jack = (numpy.where(rix))
        new_r = ra_rad[gal_jack] # applies new mask
        new_d = dec_rad[gal_jack]
        new_rand_ra = rand_ra[rand_jack]
        new_rand_dec = rand_dec[rand_jack]
        ra_jack.append(new_r)
        dec_jack.append(new_d)
        rand_ra_jack.append(new_rand_ra)
        rand_dec_jack.append(new_rand_dec)
    return (ra_jack, dec_jack, rand_ra_jack, rand_dec_jack)


# In[7]:


# function that performs the jackknifing
def jack(i,zd, ra_rad, dec_rad, rand_ra, rand_dec, unique_pix, pix_gal, pix_rand):
    ra_jack, dec_jack, rand_ra_jack, rand_dec_jack = limits(ra_rad, dec_rad, rand_ra, rand_dec, unique_pix, pix_gal, pix_rand)
    # new galay and random catalogs without one of the subsamples (does this 9 times)
    galaxy_catalog = treecorr.Catalog(ra = ra_jack[(i)], dec = dec_jack[(i)], ra_units='rad', dec_units='rad')
    rand_catalog = treecorr.Catalog(ra=rand_ra_jack[i], dec=rand_dec_jack[i], ra_units='radians', dec_units='radians')
    xi, dd = tree2(galaxy_catalog, rand_catalog, zd)
    return (xi)


# Formula for the covariance matrix was taken from Yang, Brunner,and Dolence(2012) equation 8.

# In[8]:


def errxi(xi_value, new_xi): # gets the errors for each value of xi using diag. elements of covariance matrix   
    cov_err = [] 
    for red in range (0,25):
        cov_mat = []
        for cell in range (0, xi_value[red].size):    
            sum_cov = 0
            for num in range (0,9):
                sub = xi_value[red][cell] - new_xi[red][cell][num]
                cov = math.pow(sub, 2)
                sum_cov+=cov
            cot = math.sqrt(sum_cov*8/9)
            cov_mat.append((cot))
        cov_err.append(cov_mat)
    return cov_err


# In[9]:


def cov_mat (xi_value, new_xi): # gets the entire covariance matrix    
    covi_def = [] 
    for dif in range (0,25):
        covi_final = []
        for red in range (0,xi_value[dif].size):
            covi_mat = []
            count = 0
            for cell in range (0,xi_value[dif].size):    
                sum_covi = 0
                for num in range (0,9):
                    subi = (xi_value[dif][count] - new_xi[dif][count][num])
                    subj = (xi_value[dif][red] - new_xi[dif][red][num])
                    covi = subi*subj
                    sum_covi+=covi
                tot = (sum_covi*8/9)
                covi_mat.append((tot))
                count =count+1
            covi_final.append(covi_mat)
        covi_def.append(covi_final)
    return covi_def


# In[10]:


def corr_mat(cov_err, covi_def):#This function converts the cov matrix into the correleation matrix    
    corr_def = [] 
    for cov in range (0,25):
        covinv = numpy.linalg.inv(numpy.diag(cov_err[cov]))
        corr = numpy.dot(covinv ,numpy.dot(covi_def[cov], covinv))
        corr_def.append(corr)
    return corr_def


# In[11]:


def err2log(xi_value, cov_err): # convert linear error for xi values into log error 
    log_error = [] 
    for redshift in range (0,25):
        new_error = []
        for den in range (0,len(xi_value[redshift])):
            new_error.append (0.434*cov_err[redshift][den]/xi_value[redshift][den])
        log_error.append(new_error)
    return log_error


# The following function gets the best fit paramters A and gamma using equations obtained from Numerical Recipes by Press(2002) page 781.

# In[12]:


def params (r_value, xi_value, log_error): # gets the values gor A and gamma, as well as the power law model   
    fin_A = [] 
    fin_gamma = [] 
    fin_Aerr = [] 
    fin_gerr = []
    fin_fit = [] # best fit line using values of A and gamma
    for shift in range (0,25): 
        s_sum = 0
        sx_sum = 0
        sy_sum = 0
        sxx_sum = 0
        sxy_sum = 0
        for num in range (0,len(xi_value[shift])):
            err = math.pow(log_error[shift][num],2)
            s = 1/err
            s_sum +=s
            sx = numpy.log(r_value[shift][num])/err
            sx_sum += sx
            sy = numpy.log(numpy.absolute(xi_value[shift][num]))/err
            sy_sum += sy
            sxx = sx = math.pow(numpy.log(r_value[shift][num]),2)/err
            sxx_sum += sxx 
            sxy = numpy.log(numpy.absolute(xi_value [shift][num]))*numpy.log(r_value[shift][num])/err
            sxy_sum += sxy
        delta = s_sum*sxx_sum - math.pow(sx_sum,2)
        a = (sxx_sum*sy_sum - sx_sum*sxy_sum)/delta
        b = (s_sum * sxy_sum - sx_sum*sy_sum)/delta
        err_alog = numpy.sqrt(sxx_sum/delta)
        err_a = err_alog * numpy.exp(a)/0.434
        err_b = numpy.sqrt(s_sum/delta)
        fin_A.append(numpy.exp(a))
        fin_gamma.append(1-b)
        fin_Aerr.append(err_a)
        fin_gerr.append(err_b)
        g = (fin_gamma[shift])
        slope = 1-g
        new_fit = numpy.exp((slope)*numpy.log(r_value[shift]) + numpy.log(fin_A[shift]))
        fin_fit.append(new_fit)
    return (fin_A, fin_Aerr, fin_gamma, fin_gerr, fin_fit)


# In[13]:


def tree2(galaxy_catalog, rand_catalog, zd):
    dd = treecorr.NNCorrelation(min_sep=theta_min[zd], max_sep=theta_max[zd], nbins=10, sep_units='degrees')
    rr = treecorr.NNCorrelation(min_sep=theta_min[zd], max_sep=theta_max[zd], nbins=10, sep_units='degrees')
    dr = treecorr.NNCorrelation(min_sep=theta_min[zd], max_sep=theta_max[zd], nbins=10, sep_units='degrees') 
    dd.process(galaxy_catalog)
    rr.process(rand_catalog)
    dr.process(galaxy_catalog,rand_catalog) 
    xi, varxi = dd.calculateXi(rr,dr) 
    return (xi,dd)


# In[14]:


# from config file get names of ra, dec columns and if there are any dec,ra limits
config_file = 'des40a/example/config.yaml'
config = treecorr.read_config(config_file)
DEC_COL = config['dec_col']
RA_COL = config['ra_col']
Z_COL = config['z_col']
RA_LIM = config['ra_lim']
DEC_LIM = config['dec_lim']
GET_GRAPHS = config['get_graphs']


# In[15]:


galaxy_sample = config['file_name'] # fits file with galaxy sample from config file
full_data = Table.read(galaxy_sample) #Table displaying ra,dec,redshift,etc.
z= numpy.array(full_data[Z_COL]) # array with redshift values for each galaxy
                               #beware different tables call this column differently
ra = numpy.array (full_data[RA_COL]) # array with ra values
dec = numpy.array(full_data[DEC_COL]) # array with dec values


# In[16]:


# from config file determine what the ra,dec limits are, if specified in the config file
if (RA_LIM ==True):
    ramin = config['ra_min']
    ramax = config['ra_max']
else:
    ramin = numpy.min(ra)
    ramax = numpy.max(ra)
if (DEC_LIM == True):
    decmin = config['dec_min']
    decmax = config['dec_max']
else:
    decmin = numpy.min(dec)
    decmax = numpy.max(dec)


# In[17]:


if ((RA_LIM == True) or (DEC_LIM == True)):
    mask = ((dec< decmax) & (dec> decmin) & (ra > ramin ) & (ra< ramax)) # ra and dec restrictions into mask
    data = full_data[mask] #new galaxy sample with mask applied
    z= numpy.array(data[Z_COL])
    ra = numpy.array (data[RA_COL])
    dec = numpy.array(data[DEC_COL])
else:
    data = full_data


# In[18]:


z_min, z_max = get_z(data)
theta_min, theta_max = get_theta(z_max)
xi_value = []
r_value = []
new_xi = []
#performs calculation of two point correlation function for each z-bin 
for zd in range (0,25):
   print('Computing Two Point Correlation Function for redshift bin ' + str(zd) + '...') 
   mask = (z>(z_min[zd])) & (z<(z_max[zd]))
   mask_ra = ra[mask]
   mask_dec = dec[mask]
   galaxy_catalog = treecorr.Catalog(ra = mask_ra, dec = mask_dec, ra_units='deg', dec_units='deg') 
   ra_rad = galaxy_catalog.ra
   dec_rad = galaxy_catalog.dec
   rand_catalog, rand_ra, rand_dec, unique_pix, pix_gal, pix_rand = rand(ra_rad, dec_rad, data[mask])
   xi, dd = tree2(galaxy_catalog, rand_catalog, zd)
   r = numpy.exp(dd.meanlogr)
   r_value.append(r)
   xi_value.append(xi)
   # jackknifing technique  
   newxi = [] # xi values for each jackknife run
   final_jack= [] # xi values for each angular bin in jackknife (each anguar bin has 9 values) 
   for i in range (0,9):
        jack_arr = numpy.array(jack(i,zd, ra_rad, dec_rad, rand_ra, rand_dec, unique_pix, pix_gal, pix_rand))
        newxi.append(jack_arr)
   for n in range (0,xi.size) :
       jack_xi=[] # goes through each angular bin of each jackknife and stores the xi values
       for count in range (0,9): 
           jack_xi.append(newxi[count][n])    
       final_jack.append(jack_xi)
   new_xi.append(final_jack)
   print('Completed')
xi_err = (errxi(xi_value, new_xi)) # error on each value of xi
cov_matrix = cov_mat(xi_value, new_xi) #covariance matrix
corr_matrix = corr_mat(xi_err, cov_matrix) # correlation matrix
xi_logerr = err2log(xi_value, xi_err) # log error on xi values
A_value, A_err, g_value, g_err, fit_eq = params(r_value, xi_value, xi_logerr) # final values of A, gamma with errors
                                                                              # and best fit equation


# In[19]:


if (GET_GRAPHS == True):
    fig, ax = plt.subplots(5, 5,figsize = (20,20)) #log-log scale of autocorrelation function fo each z bin
    rep = []
    for i in range(5):
        rep.append (5*i)
        for j in range(5):
            ax[i, j].plot(r_value[rep[i] + j], fit_eq[rep[i] + j], color = 'red', lw = 2)
            ax[i,j].scatter(r_value[rep[i] + j], xi_value[rep[i] + j], color='blue', s = 15)
            ax[i,j].errorbar(r_value[rep[i] + j], xi_value[rep[i] + j], yerr = (xi_err[rep[i] + j]), color='black', lw=1, ls='')              
            ax[4,j].set_xlabel(r'$\theta$ (degrees)', fontsize= 14)
            ax[i,0].set_ylabel(r'$\omega$ ($\theta$)', fontsize = 14)
            ax[i,j].xaxis.set_major_locator(plt.LinearLocator(5))
            ax[i,j].xaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
            ax[i,j].yaxis.set_major_locator(plt.LinearLocator(5))
            ax[i,j].yaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
    fig.tight_layout()
    fig.savefig('des40a/output_files/corrfunction_grid.png')
    plt.close(fig)


# In[20]:


if (GET_GRAPHS == True):    
    # graph correlation function for each z bin
    fig, ax = plt.subplots(5, 5, sharex=True, sharey=True, figsize = (20,20))
    rep = []
    for i in range(5):
        rep.append (5*i)
        for j in range(5):      
            cax = ax[i, j].imshow(corr_matrix[rep[i] + j], aspect = 'equal',interpolation = None,vmin = 0, vmax = 1, origin = 'lower')
            fig.colorbar(cax , ax=ax[i, j])
            ax[4,j].set_xlabel('bin', fontsize= 14)
            ax[i,0].set_ylabel('bin', fontsize = 14)
    fig.savefig('des40a/output_files/correlation_matrices.png')
    plt.close(fig)


# In[21]:


if (GET_GRAPHS == True):     
    sim = []
    for d in range (0,25):
        sim.append(d)
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (15,5))
    ax1.scatter (sim, g_value, color = 'blue')
    ax1.errorbar (sim, g_value, yerr = g_err,color='black', lw=1, ls='')
    ax1.set_xlabel('Redshift bin')
    ax1.set_ylabel('$\gamma$')

    ax2.scatter (sim, A_value, color = 'red')
    ax2.errorbar (sim, A_value, yerr = A_err,color='black', lw=1, ls='')
    ax2.set_xlabel('Redshift bin')
    ax2.set_ylabel('A')
    fig.savefig('des40a/output_files/parameters_graphs.png')
    plt.close(fig)


# In[22]:


#write out file
output = open ('des40a/output_files/output.info', 'w')
columnTitleRow = " ra_min     ra_max      dec_min    dec_max    zmin       zmax       A                        gamma\n"
output.write(columnTitleRow )
for i in range (0,25):
    A = ('%.7f  %.7f  %.7f  %.7f  %.7f  %.7f  %.7f +/- %.7f  %.7f +/- %.7f\n' 
         % (ramin, ramax, decmin, decmax, z_min[i], 
            z_max[i], A_value[i], A_err[i], g_value[i], g_err[i]))
    output.write(A)
output.close()

