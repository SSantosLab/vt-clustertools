#!/bin/bash

dir=$( pwd )
mkdir -p src/
mkdir -p log/

# Only create file once or it appends lines to existing file
rm mof_healpix.lst
for filename in /data/des50.a/data/annis/mof_cat_for_clusters/healpix/y1a1_gold_bpz_mof_full_area_CUT_columns_y3a2_mof/*.fits;
    do tmp=${filename#*mags_}
    temp2=${tmp%.*}
    num=${temp2##+(0)} #to remove de zeros before the ids
    echo "$num" >> mof_healpix.lst
done

#pixels=('10890' '10891' '10892' '5937')
#pixels=$( cat mof_healpix.lst )
pixels=$( head mof_healpix.lst )

for pix in ${pixels[@]}; do
    echo "python -c \"import local_afterburner_4; local_afterburner_4.nike(cluster_pix=$pix)\" ">src/afterburner_${pix}.sh
    chmod +x src/afterburner_${pix}.sh 
    csub -n 20 -o ${dir}/log/${pix}.log ./src/afterburner_${pix}.sh
done

