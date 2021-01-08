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


cat ${1} | while read line
do
    echo("ziffit.py ${FILENAME} --catalog 'ps1cat' --addfilter 'gmag' '14' '18'")
done
		 

