#!/bin/bash

PROJECTION_INDIR=
PROJECTION_OUTDIR=

OPTIND=1

while getopts "i:o:" opt; do
	case "$opt" in
		i)
			export PROJECTION_INDIR=$OPTARG
			echo "Input directory is $PROJECTION_INDIR" >&2
			;;
		o)
			export PROJECTION_OUTDIR=$OPTARG
			echo "Output directory is $PROJECTION_OUTDIR" >&2
			;;
		\?)
			echo "Invalid argument : $1" >&2
			exit 1
			;;
		:) 
			echo "Option -$OPTARG requires an argumnet." >&2
			exit 1
			;;
	esac
done

shift $((OPTIND - 1))


for f in "$PROJECTION_INDIR"/*.npy; do
	echo $f
	python preprocess_for_squeezeseg.py "$f" "$PROJECTION_OUTDIR"
done

echo "Processing complete"
