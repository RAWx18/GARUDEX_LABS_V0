# Jhakaas-Rasta

This repository contains code and data for the **Jhakaas-Rasta** project, focused on traffic monitoring, vehicle detection, traffic optimization, and more.

---

## Setup Instructions

### Python Environment Setup

To set up a virtual environment:

#### For **Windows**:
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### For **Linux & macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

---

## GPU Check (if applicable)

For machines with an NVIDIA GPU, check the GPU status with:
```bash
!nvidia-smi
```

---

## Mermaid Diagrams

For visualizing **Mermaid** diagrams, use [mermaid.live](https://mermaid.live) by pasting the content from `.txt` files in the `mermaid_files/` folder.

---

## Git Large File Handling

To remove large files from Git history:

1. **Install Git Filter Repo**:
   ```bash
   !sudo apt install git-filter-repo
   ```

2. **Remove Large Files**:
   ```bash
   git filter-repo --path <file> --invert-paths
   ```

3. **Re-add Remote**:
   ```bash
   git remote add origin <repo_url>
   ```

4. **Push Changes**:
   ```bash
   git push origin main
   ```

---

## Module Structure

Each module is in its own directory. Please update your code within these folders:

- `grid_traffic_monitor`
- `data_integration`
- `vehicle_detection_flow`
- `traffic_congestion_pred`
- `signal_optimization`
- `emergency_response`
- `system_integration_pipeline`
- `challan_communication_txn`

---

## Team Assignments

- **RYAN**: Grid Division & Traffic Monitoring, Congestion Prediction
- **RONIT**: Traffic Signal Optimization
- **PARV**: Data Integration, Emergency Response
- **AKSHITA**: Vehicle Detection & Traffic Flow Analysis
- **LAKSHYA**: Challan & Blockchain Transactions
- **KAIRVEE**: System Integration & Dashboard

---

## Note for Team Members

Whenever you make edits or updates to the code in your respective module folder, **always update the following files**:

- **`README.md`**: Document any important changes, new features, or instructions for using the updated code in your module.
- **`requirements.txt`**: If you've added new dependencies, make sure to include them in the `requirements.txt` file so others can install them.

This ensures that the project remains well-documented and easy to set up for all contributors.

---

## Notes

- **Plans**: Follow the plans in `plans/updated_plan.md` and `plans/detailed_updated_plan.md` for guidance.
- **Notebooks**: Use the `notebooks` folder for trial and error or testing.