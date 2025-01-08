# Jhakaas-Rasta

This repository contains the code and data for the **Jhakaas-Rasta** project. Below are the instructions and guidelines for setting up and managing this project.

---

## Setup Instructions

### Python Environment Setup

To create and activate a virtual environment, follow these steps:

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

---

## To Check GPU (if applicable):

If you are working on a machine with an NVIDIA GPU, you can check the GPU status using the following command:

```bash
!nvidia-smi
```

---

## Mermaid Diagrams

For viewing or editing **Mermaid** diagrams, use [mermaid.live](https://mermaid.live). Simply copy and paste the content from the `.txt` files in the `mermaid files/` folder into the tool to visualize the diagrams.

---

## Git Large File Handling

In case of any errors during pushing large files or uploading them by mistake, you can use the following commands to clean up the Git history and remove large files.

1. **Install Git Filter Repo** (if needed):
   ```bash
   !sudo apt install git-filter-repo
   ```

2. **Remove Large File from Git History**:
   Replace `datasets/output_processed.mp4` with the actual file name you want to remove.
   ```bash
   git filter-repo --path datasets/output_processed.mp4 --invert-paths
   ```

3. **Re-add the Remote Origin**:
   If the remote has been removed or is not set, re-add it:
   ```bash
   git remote add origin https://github.com/RAWx18/Jhakaas-Rasta.git
   ```

4. **Verify Remote**:
   ```bash
   git remote -v
   ```

5. **Push Changes to GitHub**:
   After making the necessary changes, you can push your clean history:
   ```bash
   git push origin main
   ```

---

### Notes

- Ensure that large files, such as `.mp4` or `.mov` files, are properly added to `.gitignore` to avoid being pushed to GitHub.
- Always check the status of your repository with `git status` before committing or pushing changes.