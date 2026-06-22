# Push this clean project to GitHub

From the extracted project folder:

```bash
cd PIConGPU_analisis_clean_project_v2

git init
git branch -M main
git remote add origin git@github.com:cemedenh/PIConGPU_analisis.git

git add .
git status
git commit -m "Rebuild clean analysis project with corrected notebooks"
git push -u origin main --force-with-lease
```

Before committing, make sure `git status` does not show `.h5`, `.bp5`, images, or checkpoint/cache folders.
