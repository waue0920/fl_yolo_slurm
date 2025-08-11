# Federated Data

This directory stores the partitioned datasets for each client, generated for federated learning experiments.

**This directory is intentionally left empty in the Git repository and will be populated by the data preparation script.**

## Setup Instructions

To generate the federated data, run the data preparation script from the project's root directory:

```bash
python src/data_prepare.py --config <path_to_your_fl_config>
```

This will create the necessary subdirectories for each client based on your configuration.
