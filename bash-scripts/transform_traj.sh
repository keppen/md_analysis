#!/usr/bin/env bash
set -euo pipefail

# Base directory containing boc-* folders
BASE_DIR=$(pwd)
TPRFILE="rerun.tpr"

# Loop over all boc-* directories
for DIR in "$BASE_DIR"/boc-*; do
    # Skip if not a directory
    [[ -d "$DIR" ]] || continue

    echo ">>> Processing directory: $DIR"
    TRAJDIR="$DIR"

    # Find all trajectories inside (adjust pattern if needed)
    NAME="full.xtc"

    echo "---- Processing trajectory: $NAME ----"

    # Step 1: Make whole
#    gmx trjconv -f "$TRAJDIR/$NAME" -s "$TRAJDIR/$TPRFILE" \
#        -ndec 3 -pbc whole -o "$TRAJDIR/tmp.xtc" <<< $'0'

    # Step 2: Center molecule
    gmx trjconv -f "$TRAJDIR/$NAME" -s "$TRAJDIR/$TPRFILE" \
        -pbc whole -o "$TRAJDIR/${NAME%.xtc}_centered.xtc" <<< $'0'

    # Clean up
    rm -f "$TRAJDIR/tmp.xtc"

    echo "---- Done: ${NAME%.xtc}_centered.xtc ----"
done

echo "✅ All trajectories processed."

