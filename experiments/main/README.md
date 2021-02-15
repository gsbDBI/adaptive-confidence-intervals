# Main Figures

This simulation script produces most figures in the paper, with the _exception_ of Figure 1 in the introduction, and its counterpart Figure 13 in Appendix A5.


**To non-Stanford members**

Each time this script is called, it selects a random configuration (e.g., a signal strength, an experiment horizon, etc) and completes a single simulation using that configuration.

In order to produce the figures in our paper, we recommend that `simulations.ipynb` be run at least $10^6$ times.



**To Stanford members with access to the Sherlock cluster**

Each time this script is called on sherlock, it selects a random configuration (e.g., a signal strength, an experiment horizon, etc) and completes 200 simulations using that configuration.

In order to produce the figures in our paper, we recommend that `simulations.ipynb` be run a few thousand times.