#!/bin/bash
DIR=$(dirname $0)

i=2

while [[ $i -le 20 ]] ; do
	j=0
	echo "Domain size: $(($i*512))"
	while [[ $j -lt 2 ]] ; do
		$DIR/cuda-hpp $(($i*512)) 1024 ${DIR}/walls.in
		((j++))
	done
	((i++))
done
