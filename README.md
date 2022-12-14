# Delayed - Indirect
This repository contains the code and experiments for **Delayed and Indirect Impacts of Link Recommendations**.

## Setting up
1. Install the dependencies with Conda. Then activate the environment.

    ```
    conda env create -f environment.yaml
    conda activate delayed-indirect
    ```

2. Run the python files `exp_delayed_effects.py`, `exp_indirect_effects.py`, and `exp_group_structure.py` to replicate our experiments. Follow `exp_visualization.ipynb` to replicate our graphs.

3. The main code is in `simulation.py`. You can play around with different experiment settings and use our plotting functions in `plotting_utils.py` to explore more interesting findings!