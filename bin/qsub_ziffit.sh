#!/bin/sh

#####################################
# exemple job script pour options GE
#####################################
#$ -o $HOME/jobs/
#$ -M m.rigault@ipnl.in2p3.fr
#$ -m be   ## envoie un email quand le job commence et termine
# Mandatory: group
#$ -P P_ztf
#
# Must Access SPS for storage
#$ -l sps=1
#
# The jobs take a lot of memory and fail on long
#$ -q long
#
#$ -l h_rss=2G

# RUNNING THE CODE

for i in $(cat $1); do
    qsub /sps/lsst/users/rigault/anaconda3/bin/ziffit.py ${1} --catalog 'ps1cat' --addfilter 'gmag' '14' '18'
done


