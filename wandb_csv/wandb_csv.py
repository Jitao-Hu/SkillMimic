import pandas as pd
import wandb
import json
import os
from datetime import datetime

# 1. Setup basic info
entity = "xinyuye2018-university-of-waterloo"
project = "SkillMimic"
run_id = "sa3agp84"

# 2. Create directory with timestamp (YYYYMMDD_HHMMSS)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"wandb_export_{timestamp}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# 3. Initialize API and fetch run data
api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")

print(f"Fetching data from run: {run.name}...")

# 4. Export full history metrics to CSV
# Set samples high enough to capture all data points
history_df = run.history(samples=100000)
csv_path = os.path.join(output_dir, "full_history.csv")
history_df.to_csv(csv_path, index=False)
print(f"Trend data saved to: {csv_path}")

# 5. Export hyperparameter config to JSON
config = run.config
config_path = os.path.join(output_dir, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"Hyperparameter config saved to: {config_path}")

# 6. Export final summary to JSON
summary = run.summary._json_dict
summary_path = os.path.join(output_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Final summary data saved to: {summary_path}")

print("\nExport complete! You can now send the files in this folder to me.")