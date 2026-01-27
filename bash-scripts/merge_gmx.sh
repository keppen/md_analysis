#! /bin/bash
#set -euo pipefail

ROOT="/media/szatko/Projects/side_chain"
TRAJDIR="$ROOT/sim/$1"

TOPFILE="$TRAJDIR/classic.top"
TPRFILE="$TRAJDIR/0-classic-0.tpr"
GROFILE="$TRAJDIR/0-classic-0.gro"
NDXFILE="$TRAJDIR/index.ndx"

#gmx make_ndx -f "$TPRFILE" -o "$NDXFILE" <<< "q\n"


PATTERN="boc-[a-z]*[0-9]*"

if [[ $TRAJDIR =~ $PATTERN ]]
then
  SYSTEMNAME="$BASH_REMATCH"
else
  echo "Did not accessed system name"
  exit 1
fi

#ANALISISDIR="$ROOT/analysis/$SYSTEMNAME"
ANALISISDIR="$ROOT/analysis/$SYSTEMNAME"

if [[ ! -d $ANALISISDIR ]]
then
  echo "Analysis dir does not exists."
  mkdir "$ANALISISDIR" -p
fi

cd "$ANALISISDIR"

# merge trajectories and update timestep
find "$TRAJDIR" -name "0-classic-*.trr" | sort | xargs gmx trjcat -o tmp.xtc -cat -f
gmx trjconv -f "tmp.xtc" -o "full.xtc" -timestep "$TS"

FULLTRR="$ANALISISDIR/full.xtc"

rm "tmp.xtc"

# cut trajectory
# gmx trjconv -f "full.xtc" -o "dt20.xtc" -dt $DT

ISODIR="$ANALISISDIR/iso"

if [[ ! -d $ISODIR ]]
then
  echo "Analysis dir does not exists."
  mkdir "$ISODIR" -p
fi

cd "$ISODIR"

for TRR in "$TRAJDIR"/*trr
do
    NAME=$(basename "$TRR" .trr)
    # isolated molecule
    # "2" or "2 \n 2" are the selection of groups defined in index file
    gmx trjconv -f "$TRAJDIR/$NAME" -s "$TPRFILE" -ndec 3 -pbc whole -o "tmp.xtc" <<< $'2'
    gmx trjconv -f "tmp.xtc" -s "$TPRFILE" -pbc mol -center -o "$NAME.xtc" <<< $'2 \n 2'

    rm "tmp.xtc"

    # yes c | gmx eneconv -o iso.edr -settime \
    #        -f $(find "$TRAJDIR" -name "0-classic-*.edr" | sort)
done
