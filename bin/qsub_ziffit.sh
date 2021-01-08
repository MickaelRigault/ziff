#!/bin/sh

#####################################
# exemple job script pour options GE
#####################################

# RUNNING THE CODE

for i in $(cat $1); do
    qsub -P P_ztf -l sps=1 -q long -l h_rss=2G /sps/lsst/users/rigault/anaconda3/bin/ziffit.py ${i} --catalog 'ps1cat' --addfilter 'gmag' '14' '18'
done


