# Additional figures

The notebook `intro_example.ipynb` plots Figure 1 in the introduction and Figure 13 in Appendix A5.


### Reproducing figures (General).

**Step 1**

Open `intro_example_simulations.ipynb` on jupyter lab or jupyter notebook and run it. Or, on Terminal:
```
python intro_example_simulations.py
```
The results will be stored in folder `results/`

**Step 2**

Open `intro_example_plots.ipynb` on jupyter lab or jupyter notebook and run it. Plots will be stored in folder `figures`.



#### To Stanford users with access to Sherlock

Users with access to Stanford's Sherlock cluster can run simulations above in parallel via:
```
sbatch [-p PARTITION] jobfile.job
```
