ROOT="/media/szatko/Projects/side_chain"


cd $ROOT

for DIR in slurm*
do 
  SYSTEM=${DIR//slurm-classic-}

  cd "$ROOT/analysis/$SYSTEM/clustering"

  for FILE in *pdb
  do
    grep LIG $FILE -vw > "iso-${FILE}"
  done

done

