USE mermaid.live
for mermaid files

PYTHON ENV:
```
python -m venv .venv # windows
.venv\Scripts\activate # windows

python3 -m venv .venv # linux & macos
source .venv/bin/activate # linux & macos
```

TO CHECK GPU:
```
!nvidia-smi
```

Incase of any error of pushing and uploading any large file by mistake :
```
!sudo apt install git-filter-repo
git filter-repo --path datasets/output_processed.mp4 --invert-paths
git remote add origin https://github.com/RAWx18/Jhakaas-Rasta.git
git remote -v
git push origin main
```