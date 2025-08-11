# Federated Data

This directory stores the partitioned datasets for each client, generated for federated learning experiments.

**This directory is intentionally left empty in the Git repository and will be populated by the data preparation script.**

## Setup Instructions

To generate the federated data, run the data preparation script from the project's root directory:

```bash
python3 src/data_prepare.py     --dataset-name "kitti"     --num-clients "4"
```

This will create the necessary subdirectories for each client based on your configuration.
