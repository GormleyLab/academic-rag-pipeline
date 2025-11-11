# Quick Start Guide

Get up and running with the Academic RAG Pipeline in 10 minutes!

## 1. Install Dependencies (2 min)

```bash
pip install -r requirements.txt
```

## 2. Set Up Environment (1 min)

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
# CROSSREF_EMAIL=your.email@edu
```

## 3. Configure PDF Path (30 sec)

Edit `config/config.yaml`:
```yaml
pdf_library_path: "/path/to/your/pdfs"  # Change this line
```

## 4. Initialize Database (varies by library size)

```bash
python scripts/initial_setup.py
```

Expected: ~2 minutes per paper, ~$0.015 per paper

## 5. Configure Claude Desktop (2 min)

Edit Claude Desktop config:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add:
```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": ["/FULL/PATH/TO/academic-rag-pipeline/src/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/FULL/PATH/TO/academic-rag-pipeline",
        "OPENAI_API_KEY": "sk-your-key-here",
        "CONFIG_PATH": "/FULL/PATH/TO/academic-rag-pipeline/config/config.yaml"
      }
    }
  }
}
```

Replace `/FULL/PATH/TO/` with your actual path!

## 6. Restart Claude Desktop

Quit and reopen Claude Desktop app.

## 7. Test It! (30 sec)

In Claude Desktop, try:
```
Search my papers for "machine learning"
```

You should see results from your database!

## Common Commands

### Search for papers
```
Find papers about [topic] from the last 3 years
```

### Add a paper
1. Drag PDF into Claude Desktop chat
2. Say: "Add this paper to my database"

### Generate bibliography
```
Generate a .bib file with Smith2024, Jones2023, Chen2022
```

### Get paper details
```
Show me information about the Wang2024 paper
```

### Database stats
```
How many papers are in my database?
```

## Troubleshooting

**MCP server not showing up?**
- Check Claude Desktop logs
- Verify Python path is correct (use absolute paths)
- Make sure you restarted Claude Desktop

**"OpenAI API Key Not Found"?**
- Check `.env` file has your key
- Check Claude Desktop config has the key
- No quotes needed around the key

**Papers not processing?**
- Check `data/logs/initial_setup.log`
- Make sure PDF path is correct
- Verify PDFs are not encrypted

## Next Steps

- Read [docs/setup.md](docs/setup.md) for detailed setup
- Check [PRD.md](PRD.md) for all features
- See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for what's implemented

---

**Need help?** Check the logs in `data/logs/` for error messages.
