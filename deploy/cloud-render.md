# deploy/cloud-render.md — Step 13: Render Deployment

Imported by @deploy/cloud.md.

---

## Step 13 — Deploy on Render

> ⚠️ **AGENT RULE — READ FIRST:**
> - **DO NOT** use `render` CLI, `render login`, `render deploy`, or any Render CLI commands.
> - **DO NOT** ask for a Render API key or attempt API-based deployment.
> - **DO NOT** try to automate the Render dashboard — it requires a browser.
> - Your job is: (1) create `render.yaml`, (2) push it to GitHub, (3) print the manual steps for the user.
> - Stop after step 13e. The user will complete the browser steps themselves.

Perform this step after the Dockerfile is pushed to GitHub (Step 12).

### 13a — Create `render.yaml`
Create `render.yaml` at the project root:
```yaml
services:
  - type: web
    name: <project-name>
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.0"
```

### 13b — Push render.yaml to GitHub
```bash
git add render.yaml
git commit -m "Add render.yaml for Render deployment"
git push origin main
```

### 13c — Print manual deploy instructions for the user
Print this message to the user **exactly** (replace `<username>` and `<project-name>` from `.ml_config.json`):

```
✅ render.yaml is ready and pushed to GitHub.

To go live on Render (free, ~2 minutes):
  1. Visit https://render.com → sign in with GitHub
  2. Click New + → Web Service
  3. Connect repo: <username>/<project-name>
  4. Render auto-detects render.yaml → click Create Web Service
  5. Your API will be live at: https://<project-name>.onrender.com

Once live, test it:
  curl https://<project-name>.onrender.com/health
```

### 13d — Create `deployment_guide.md`
Document the following in `docs/deployment_guide.md`:
- Prerequisites (files needed before deploying)
- 5-step Render deploy walkthrough with exact settings
- All API endpoint descriptions
- Test-it-live curl commands (health, predict, batch)
- Run locally instructions
- Input field reference table
- Free tier cold-start note

### 13e — Push deployment guide to GitHub
```bash
git add docs/deployment_guide.md
git commit -m "Add deployment guide"
git push origin main
```

**Stop here.** The user will open their browser to complete the Render dashboard steps.
