# Push this clean project to GitHub

From your PC or from Jupyter terminal, after extracting this zip:

```bash
cd PIConGPU_analisis_clean_project

git init
git branch -M main
git remote add origin git@github.com:cemedenh/PIConGPU_analisis.git

git add .
git status
git commit -m "Rebuild clean analysis project"
git push -u origin main --force-with-lease
```

Use `--force-with-lease` only if you want this clean folder to replace the current GitHub content.

If you prefer not to overwrite history, do this instead:

```bash
git checkout -b clean-rebuild
git push -u origin clean-rebuild
```

Then merge the branch on GitHub.

## Do not add generated data

Do not commit:

```text
*.bp5
*.h5
*.png
simOutput/
openPMD/
.ipynb_checkpoints/
__pycache__/
```
