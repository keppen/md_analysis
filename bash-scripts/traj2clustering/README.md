## ./merge_gmx.sh ##

# Overview #

This script processes molecular dynamics trajectories generated with GROMACS. It:

 - Merges trajectory segments (.trr)
 - Converts them into a single .xtc trajectory
 - Generates per-frame isolated-molecule trajectories
 - Organizes outputs into a structured analysis directory

== Usage ==

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

`simulation_name` have hadcoded format `boc-[a-z]*[0-9]*`. E. g. `<simulation_name>`: boc-ls4-acn-classic -> `<system_name>`: boc-ls4

== Output structure ==

Output has hardcoded path 

`/media/szatko/Projects/side_chain/analysis/<system_name>/`

```bash
analysis/<system_name>/
├── full.xtc        # merged and time-corrected trajectory
├── iso/
│   ├── *.xtc       # per-trajectory isolated molecule trajectories
```

# ./clustering_gmx.sh # 

