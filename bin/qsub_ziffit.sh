#!/bin/sh

#####################################
# exemple job script pour options GE
#####################################


# RUNNING THE CODE

for i in $(cat $1); do
    qsub -P P_ztf -l sps=1 -q long -l h_rss=2G -o $HOME/jobs/ziffit/ /sps/lsst/users/rigault/anaconda3/bin/ziffit.py ${i} --catalog 'ps1cal' --addfilter 'gmag' '14' '18'
done


