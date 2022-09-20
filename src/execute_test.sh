#!/bin/bash
DIR=$(dirname $0)

i=1

while [[ $i -le 12 ]] ; do
	j=0
	echo "Number of threads: ${i}"
	while [[ $j -lt 2 ]] ; do
		OMP_NUM_THREADS=$i $DIR/omp-hpp 2048 1024 ${DIR}/walls.in
		((j++))
	done
	((i++))
done
