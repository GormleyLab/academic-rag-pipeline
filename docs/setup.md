# Setup Guide - Academic RAG Pipeline

This guide will walk you through setting up the Academic RAG Pipeline from scratch.

## Prerequisites

- Python 3.10 or higher
- OpenAI API account with API key
- Claude Desktop app installed
- A local library of research papers (PDFs)

## Step 1: Install Dependencies

```bash
cd academic-rag-pipeline
pip install -r requirements.txt
```

**Note:** This may take a few minutes as it installs all required packages including Docling, LanceDB, and OpenAI.

## Step 2: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   OPENAI_API_KEY=sk-your-actual-api-key-here
   CROSSREF_EMAIL=your.email@university.edu
   ```

## Step 3: Configure the Application

1. Edit `config/config.yaml`:
   ```yaml
   # Change this to your PDF library path
   pdf_library_path: "/Users/yourname/Documents/Papers"
   ```

2. Review other settings (optional):
   - `max_chunk_tokens`: Size of text chunks (default: 1000)
   - `batch_size`: Number of embeddings per API call (default: 100)
   - `lancedb_path`: Where to store the vector database

## Step 4: Run Initial Setup

Process your PDF library and create the vector database:

```bash
python scripts/initial_setup.py
```

This will:
- Scan your PDF directory for all PDF files
- Extract text and metadata from each paper
- Generate embeddings using OpenAI API
- Store everything in LanceDB

**Expected time:** ~2 minutes per paper (depends on paper length and API speed)

**Expected cost:** ~$0.015 per paper for embeddings

## Step 5: Configure Claude Desktop

1. Locate your Claude Desktop configuration file:
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the MCP server configuration. Edit the file and add:

```json
{
  "mcpServers": {
    "academic-rag": {
      "command": "python",
      "args": [
        "/absolute/path/to/academic-rag-pipeline/src/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/academic-rag-pipeline",
        "OPENAI_API_KEY": "your-api-key-here",
        "CONFIG_PATH": "/absolute/path/to/academic-rag-pipeline/config/config.yaml"
      }
    }
  }
}
```

**Important:** Replace `/absolute/path/to/academic-rag-pipeline` with the actual full path to your project directory.

3. **Restart Claude Desktop** for changes to take effect.

## Step 6: Verify Installation

1. Open Claude Desktop

2. Look for the MCP server indicator (usually a tool icon or server status)

3. Try a test query:
   ```
   Search my papers for "machine learning"
   ```

4. You should see results from your database!

## Troubleshooting

### MCP Server Not Showing Up

1. Check Claude Desktop logs:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`

2. Verify Python path in config is correct

3. Test the MCP server manually:
   ```bash
   python src/mcp_server.py
   ```

### "OpenAI API Key Not Found"

- Make sure `OPENAI_API_KEY` is set in both:
  - `.env` file
  - Claude Desktop config JSON

### "Config file not found"

- Verify `config.yaml` path is correct
- Use absolute paths, not relative paths

### Papers Not Processing

- Check `data/logs/initial_setup.log` for errors
- Common issues:
  - Corrupted PDFs (will be skipped)
  - Encrypted PDFs (not supported)
  - Permission issues

## Adding More Papers Later

After initial setup, you can add papers in two ways:

### Option 1: Via Claude Desktop (Recommended)

1. Open Claude Desktop
2. Drag a PDF file into the chat
3. Say: "Add this paper to my database"

### Option 2: Re-run Setup Script

Add new PDFs to your library directory and run:
```bash
python scripts/initial_setup.py
```

It will only process new papers (duplicates are automatically detected).

## Next Steps

- Read [usage.md](usage.md) for detailed usage examples
- See [troubleshooting.md](troubleshooting.md) for common issues
- Check [PRD.md](../PRD.md) for full feature documentation

## Cost Management

**Initial Setup:**
- 1000 papers ≈ $15 one-time cost

**Ongoing Usage:**
- Queries: ~$0.0001 per query
- Adding papers: ~$0.015 per paper

**Tips to reduce costs:**
- Process papers in batches
- Use smaller `batch_size` if hitting rate limits
- Monitor usage in `data/logs/`

## Security Notes

- API keys are stored in `.env` (not committed to git)
- All PDFs stay local on your machine
- Only text chunks are sent to OpenAI for embedding
- LanceDB runs entirely locally (no cloud storage)

## Getting Help

- Check the logs: `data/logs/`
- Review the PRD for detailed specifications
- File issues on the project repository

---

**Happy researching!** 📚✨
