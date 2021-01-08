#!/bin/sh

#####################################
# exemple job script pour options GE
#####################################

# RUNNING THE CODE

for i in $(cat $1); do
    qsub /sps/lsst/users/rigault/anaconda3/bin/ziffit.py ${1} --catalog 'ps1cat' --addfilter 'gmag' '14' '18'  -o $HOME/jobs/ -M m.rigault@ipnl.in2p3.fr -m be -P P_ztf -l sps=1 -q long -l h_rss=2G
done


