# Main Figures

This simulation script produces most figures in the paper, with the _exception_ of Figure 1 in the introduction, and its counterpart Figure 13 in Appendix A5.


### Reproducing figures (General).

**Step 0 [Optional]**

This is required to reproduce our Figure 11, but can be skipped if there is no interest in comparing to w-decorrelation. Run
```
python compute_wdecorrelation_lambda.py
```
to precompute bias-variance trade-off parameters for the W-decorrelation method. 

**Step 1: Simulation**

Open `simulations.ipynb` on jupyter lab or jupyter notebook and run it. Or, on Terminal:
```
python simulations.py
```
The results will be stored in folder `results/`.


**Step 2: Aggregation**

Once the results are ready, they can be aggregated via the following script.
```
python aggregate.py
```


**Step 3: Plotting**

Open `plots.ipynb` on jupyter lab or jupyter notebook and run it. Plots will be stored in folder `figures/`.



#### To Stanford users with access to Sherlock

Users with access to Stanford's Sherlock cluster can run simulations above in parallel via:
```
sbatch [-p PARTITION] jobfile-wdecorr.job
sbatch [-p PARTITION] jobfile.job
```
Other users can ignore these files.