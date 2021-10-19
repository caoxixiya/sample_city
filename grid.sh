#!/bin/bash
declare -i i=0
while ((i<20))
do
	nohup python sample_city_vec.py --subset_id $i > tmp$i 2>&1 & 
	let i++
done

