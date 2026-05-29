# How to Run the ML Pipeline Template

## Step 1 — Check Python
```bash
python3 --version
```

## Step 2 — Download the bootstrap script
```bash
curl -O https://raw.githubusercontent.com/ramleo/builds_bootstrap/main/bootstrap.py
```

## Step 3 — Run the bootstrap
```bash
python3 bootstrap.py
```

## Step 4 — Enter the folder
```bash
cd builds_bootstrap
```

## Step 5 — Start the wizard
```bash
./start.sh
```

Choose **3** (Claude Code, default) and answer the prompts.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3: command not found` | Install Python 3.9+ from python.org |
| `claude: command not found` | Run `npm install -g @anthropic-ai/claude-code` |
| `Permission denied: ./start.sh` | Run `chmod +x start.sh` first |
| Dataset not found | Copy your `.csv` into `data/` and tell Claude the filename |
