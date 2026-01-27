You should firts do the renaming of the atomanames in pdg of you isolated polymer
do it with
python3 ./polymer_manager.py path/to/pdb. This will create a lot of junk, but the final output will have "NAMED" word in the name.
this will produce new pdb with new atom names in it.
Look also at the console output, ther will be backbone indexes that can be used in clustering. 

Before clustering, trajectories has to be merged. The script is at ./bash-scripts/merge_gmx.sh
The clustering scritp is at ./bash-scripts/clustering_gmx.sh

As for analysis, you can rely on ./SASA.py ./dihedrals_analysis.py ./hydrogen_bonds_matrix.py

Usage:
./dihedrals_analysis.py NAMED_structure.pdb multiple_structure.pdb

multiple_structure is obstained in the clustering.

./SASA.py path/to/some/trajectory  NAMED_structure.pdb

./hydrogen_bonds_matrix.py path/to/some/trajectories NAMED_structure.pdb path/to/appriopriate/top/file path/to/appriopriate/top/file path/to/appriopriate/top/file path/to/appriopriate/top/file

I did not check it but if you select trajectory with solvent molecules then topology also should have solvent molecules.


