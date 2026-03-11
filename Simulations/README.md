
Content here includes: 
1. System-prep-for-sim folder which contains all the parameter and initial coordinates and how these systems were created to reproduce the simulations from the docked poses.

2. DBSCAN-clustering.in is the input script for CPPTRAJ to pergorm the DBSCAN hierarchichal algorithm to classify simulation states.

3. DBSCAN_representative_39XX.pdb are the representative poses that emerged from the simulations of the two main hits.

4. FINAL_DECOMP_MMGBSA_39XX.dat contains the energies decomposed by residue from the analyzed simulations for the two hit systems.

5. MMGBSA calculations were performed in amber using the mmpbsa.in script following the recipe in mmpbsa_recipe.txt

6. contacts.py is the python code to extract the contacts within 5.5 A distance around the ligand
