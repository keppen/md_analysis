#! /bin/bash

# TS=2
TRAJDIR="$ROOT/sim/$1" # $1 name of the system
CT=$2  # cut-off for RMSD clustering. It is study/system specyfic
DT=$3 # delay time, select a frame every dt interaval for clustering
ROOT="/media/szatko/Projects/side_chain"

TOPFILE="$TRAJDIR/classic.top"
TPRFILE="$TRAJDIR/0-classic-0.tpr"
GROFILE="$TRAJDIR/0-classic-0.gro"
NDXFILE="$TRAJDIR/index.ndx" # you have to generate it and modify it. There you specify which atom indexes you want to be submitted to RMSD clustering

PATTERN="boc-[a-z]*[0-9]*" # regex to fing my specyfic system names

if [[ $TRAJDIR =~ $PATTERN ]]
then
  SYSTEMNAME="$BASH_REMATCH"
else
  echo "Did not accessed system name"
  exit 1
fi

ANALISISDIR="$ROOT/analysis/$SYSTEMNAME"

if [[ ! -d $ANALISISDIR ]]
then
  echo "Analysis dir does not exists."
  mkdir "$ANALISISDIR" -p
fi

cd "$ANALISISDIR"


FULLTRR="$ANALISISDIR/full.xtc"

ISOTRR="$ANALISISDIR/iso.xtc"


#CLUSTERDIR="$ANALISISDIR/clustering"
#CLUSTERDIR="$ANALISISDIR/CT$CT"
CLUSTERDIR="$ANALISISDIR/CT$CT-DT$DT"

if [[ ! -d $CLUSTERDIR ]]
then
  echo "Cluster dir does not exists."
  mkdir "$CLUSTERDIR" -p
fi

cd "$CLUSTERDIR" 

# TIME_STEP=$(python3 -c "print(int( $DT * 2000))")
#CT=0.2
#DT=50 # in ps
STOP_TIME=99999999999 # in ps
TRJ_NUMBER=$(python3 -c "print(int( $STOP_TIME / 10000 ))")

# OUTPUT="$DIR/${TRJ_NUMBER}x10nsx${TIME_STEP}psxOPLS"

echo DT: "$DT" STOP TIME: "$STOP_TIME" TRAJ COUNT: "$TRJ_NUMBER"

gmx cluster \
  -n      "$NDXFILE" \
  -s      "$TPRFILE" \
  -f      "$FULLTRR" \
  -g      "${SYSTEMNAME}.log" \
  -o      "${SYSTEMNAME}-cls.xpm" \
  -dist   "${SYSTEMNAME}-rmsd-dist.xvg" \
  -sz     "${SYSTEMNAME}-size.xvg" \
  -tr     "${SYSTEMNAME}-trans.xpm" \
  -ntr    "${SYSTEMNAME}-trans.xvg" \
  -clid   "${SYSTEMNAME}-id.xvg" \
  -cl     "${SYSTEMNAME}.pdb" \
  -clndx  "${SYSTEMNAME}-ndx.ndx" \
  -cutoff "$CT" \
  -method gromos \
  -wcl    100 \
  -dt     "$DT" \
  -e      "$STOP_TIME" \
  -nst    1  <<< $'3\n2'
    # "2" or "3 \n 2" are the selection of groups defined in index file. First is the what should be an input atom group and the second is the output group. Be carefull here!
