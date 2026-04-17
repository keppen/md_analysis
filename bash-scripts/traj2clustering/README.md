# ./merge_gmx.sh #

# Overview #

This script processes molecular dynamics trajectories generated with GROMACS. It:

 - Merges trajectory segments (.trr)
 - Converts them into a single .xtc trajectory
 - Generates per-frame isolated-molecule trajectories
 - Organizes outputs into a structured analysis directory

# Usage #

```bash
./merge_gmx.sh <simulation_name>
# example
./merge_gmx.sh boc-ls4-acn-classic
```

Where <simulation_name> is direcotry under:

```bash
/media/szatko/Projects/side_chain/sim/ # Path is hard coded
```

The structure of `sim` directory:

```bash 
/media/szatko/Projects/side_chain/
├── sim/
│   └── <simulation_name>/
│       ├── classic.top
│       ├── 0-classic-0.tpr
│       ├── 0-classic-0.gro
│       ├── index.ndx
│       ├── 0-classic-*.trr
```

**Note**

`simulation_name` have hadcoded format `boc-[a-z]*[0-9]*`. System name is read from <simulation_name>. E.g.: `<simulation_name>: boc-ls4-acn-classic` -> `<system_name>: boc-ls4`

# Output structure #

Output has hardcoded path 

`/media/szatko/Projects/side_chain/analysis/<system_name>/`

```bash
analysis/<system_name>/
├── full.xtc        # merged and time-corrected trajectory
├── iso/
│   ├── *.xtc       # per-trajectory isolated molecule trajectories
```

# ./clustering_gmx.sh # 

# Overview #

This script performs RMSD-based clustering on molecular dynamics trajectories
using GROMACS. It operates on a precomputed trajectory (full.xtc) and produces
clustering outputs for structural analysis.

# Usage #

```bash
./clustering_gmx.sh <simulation_name> <cutoff> <dt>
# example
./clustering_gmx.sh boc-ls4-acn-classic 0.2 20
```

Where <simulation_name> is direcotry under:

```bash
/media/szatko/Projects/side_chain/sim/ # Path is hard coded
```

Expected input structure:

```bash 
/media/szatko/Projects/side_chain/
├── sim/
│   └── <simulation_name>/
│       ├── classic.top
│       ├── 0-classic-0.tpr
│       ├── 0-classic-0.gro
│       ├── index.ndx
├── analysis/
│   └── <system_name>/
│       ├── full.xtc   # required input trajectory
```

**Note**

`simulation_name` have hadcoded format `boc-[a-z]*[0-9]*`. System name is read from <simulation_name>. E.g.: `<simulation_name>: boc-ls4-acn-classic` -> `<system_name>: boc-ls4`

# Output structure #

Output has hardcoded path 

`analysis/<system_name>/CT<CT>-DT<DT>/`

```bash
CT<CT>-DT<DT>/
├── <system>.log             # clustering log
├── <system>-cls.xpm         # cluster map
├── <system>-rmsd-dist.xvg   # RMSD distribution
├── <system>-size.xvg        # cluster sizes
├── <system>-trans.xpm       # transition matrix (visual)
├── <system>-trans.xvg       # transition data
├── <system>-id.xvg          # cluster assignment vs time
├── <system>.pdb             # cluster representatives
├── <system>-ndx.ndx         # cluster index groups
```

