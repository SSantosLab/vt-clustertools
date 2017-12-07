# Setup conda enviroment
export CONDA_DIR=/cvmfs/des.opensciencegrid.org/fnal/anaconda2/
export PATH=$CONDA_DIR/bin:$PATH

# Gaussian mix model of Sklearn modified to accept weights 
export PYTHONPATH=/home/s1/bwelch/new_sklearn/:/home/s1/bwelch/esutil/lib/python2.7/site-packages:$PYTHONPATH

# Enable standard python lybraries
source activate des17a 

# Add csub folder
export PATH=~kadrlica/bin:$PATH 

# Add joblib folder
export PYTHONPATH=$PYTHONPATH:/data/des41.a/data/marcelle/joblib-0.9.0b4

