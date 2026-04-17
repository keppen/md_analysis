# ./plot_size.py # 
# Overview #

This script reads RMSD clustering outputs (from GROMACS) and generates stacked
bar plots showing cluster population distributions across multiple systems. It
aggregates minor clusters into an “Others” category and orders systems by
stereochemistry.

# Usage #
`python ./plot_size.py <system1> <system2> ...`

Each `<system>` must follow this structure:


`./<system>/RMSD/<system>-size.xvg`

Examples
```bash
python ./plot_size.py boc-a4 boc-pgs4 boc-pgsr4 
python ./plot_size.py boc-pg*   # Wildcards work. boc-pg* analysis direcotries have to exist
```

# Output #
`cluster-stacked.png`

High-resolution stacked bar chart:
X-axis: systems (mapped to readable names)
Y-axis: population (%)
Bars: cluster contributions (top clusters + “Others”)

# ./energies_plot.py #

# Overview #

This script processes energy output files (ener.xvg) from GROMACS and computes
average energy components for multiple systems. It generates separate bar plots
for each energy term across systems.

# Usage #
`python script.py`

Run in a directory containing:

`./boc-pg*/ener.xvg # Wildcarded names are hard coded.`

Each system directory must include an ener.xvg file.

# Output #

Four high-resolution bar plots:

```bash
Electrostatic_-_Short_Range_energy_bar.png
Dispersive_-_Short_Range_energy_bar.png
Electrostatic_-_1-4_Bonded_energy_bar.png
Dispersive_-_1-4_Bonded_energy_bar.png
```

Each plot:

X-axis: systems (mapped to readable names)
Y-axis: average energy (kJ/mol)
