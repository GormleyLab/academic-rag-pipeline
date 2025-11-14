# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic RAG Pipeline - A locally-hosted RAG system for academic research paper management. Combines Docling (PDF processing), LanceDB (vector storage), OpenAI embeddings, and MCP (Model Context Protocol) to provide intelligent paper search and citation management through Claude Desktop.

**Key Constraint**: Privacy-first architecture - all PDFs and databases stored locally. Only embedding generation calls external API (OpenAI).

**Output Structure**: For each processed PDF, the system generates:
- Vector embeddings stored in LanceDB (`data/lancedb/`)
- Individual BibTeX file in `data/bibs/` (named after the PDF file)

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: Add OPENAI_API_KEY and CROSSREF_EMAIL

# Configure PDF library path
# Edit config/config.yaml: Set pdf_library_path to your PDF directory

# Initialize database (processes all PDFs)
python scripts/initial_setup.py
```

## Testing the MCP Server

```bash
# Run MCP server directly (for debugging)
python src/mcp_server.py

# The server communicates via stdio (MCP protocol)
# In production, it's launched by Claude Desktop
```

## Architecture Overview

### Data Flow Pipeline

1. **PDF Ingestion** → `DocumentProcessor` (Docling) → Markdown + Chunks
2. **Metadata Extraction** → `MetadataExtractor` → pdf2bib/DOI → CrossRef API → BibTeX
3. **Embedding Generation** → `EmbeddingGenerator` (OpenAI) → 3072-dim vectors (batched)
4. **Storage** → `VectorStore` (LanceDB) → Persistent local DB
5. **Query** → User question → Embedding → Vector similarity search → Results

### Component Responsibilities

**`src/mcp_server.py`** - MCP server entry point
- Lazy-loads all components on first tool call
- Implements 6 async tool handlers for Claude Desktop
- Configuration loaded from `config/config.yaml` + environment variables
- All components are module-level singletons to avoid re-initialization

**`src/document_processor.py`** - PDF processing
- Uses Docling's `DocumentConverter` for PDF parsing
- Docling's `HybridChunker` for semantic chunking (respects token limits)
- Creates `DocumentChunk` dataclass instances with metadata (section, page, element type)
- Element types: paragraph, table, figure, equation

**`src/metadata_extractor.py`** - Citation metadata
- **Strategy pattern** for metadata extraction (tries in order):
  1. **pdf2bib** → Extracts DOI/arXiv/PMID directly from PDF → Fetches metadata (NEW - most reliable)
  2. DOI → CrossRef API (BibTeX endpoint) - from text extraction
  3. arXiv ID → arXiv API (XML parsing)
  4. PubMed ID → PubMed API (placeholder - not fully implemented)
  5. PDF metadata (PyMuPDF)
  6. Document text parsing (fallback)
- Returns `PaperMetadata` dataclass with `ExtractionMethod` enum (includes PDF2BIB)
- Handles BibTeX key collision with letter suffixes (Smith2024a, Smith2024b)
- **pdf2bib integration**: Uses the pdf2doi library to extract identifiers directly from PDF binary content, which is more reliable than text-based regex extraction

**`src/embeddings.py`** - OpenAI integration
- Batch processing (default 100 texts/batch) with retry logic
- Exponential backoff on API errors (max 3 retries)
- Cost estimation: $0.00013 per 1K tokens for text-embedding-3-large
- Returns `EmbeddingResult` with token count and cost tracking

**`src/vector_store.py`** - LanceDB operations
- **Schema**: 23 fields including paper metadata, chunk metadata, vector (3072-dim)
- **Duplicate detection**: SHA256 hash of PDF content (first 16 chars = paper_id)
- **Search filtering**: section name, publication year, tags
- LanceDB doesn't support in-place updates easily - updates require read/modify/re-add
- All chunks for a paper share same paper-level metadata (denormalized for query performance)

**`src/bibliography.py`** - BibTeX management
- Uses `pybtex` library for parsing/generation
- Supports citation styles: inline `[Smith2024]`, APA, MLA
- Can extract citation keys from LaTeX documents (regex patterns for `\cite{}`, `\citep{}`, `\citet{}`)

**`src/utils.py`** - Helper utilities
- DOI/arXiv/PMID regex extraction from text (fallback methods - pdf2bib is primary)
- BibTeX key generation with collision handling
- File hashing (SHA256) for duplicate detection
- Logger setup with file + console output
- Text cleaning, filename sanitization, timestamp generation
- `save_bibtex_file()` - Saves individual .bib files to `data/bibs/` directory

### Configuration Hierarchy

1. `config/config.yaml` - Base configuration
2. Environment variables (`.env`) - Override YAML (takes precedence)
3. `CONFIG_PATH` env var - Can override config file location

**Critical paths in config**:
- `pdf_library_path` - Where PDFs are stored
- `lancedb_path` - Vector database location (default: `./data/lancedb`)
- `bibs_output_path` - Individual BibTeX files location (default: `./data/bibs`)
- `openai_api_key` - Required for embeddings
- `crossref_email` - Polite CrossRef API usage (optional but recommended)

### MCP Tools (6 implemented)

Each tool is an async function in `mcp_server.py`:
1. `search_papers` - Query vector DB, returns formatted results with URLs
2. `add_paper_from_file` - Full pipeline: process → extract → embed → store → save .bib file
3. `generate_bibliography` - Retrieve papers by keys, write .bib file
4. `get_paper_details` - Fetch paper metadata by BibTeX key
5. `database_stats` - Aggregate statistics (paper count, year distribution)
6. `list_recent_papers` - Sort by `date_added` field

**Note**: Tools 2 automatically saves individual .bib files to `data/bibs/` for each processed PDF.

### Key Dataclasses

```python
# src/document_processor.py
DocumentChunk(text, chunk_index, section_title, section_hierarchy, page_number, element_type, source_document)

# src/metadata_extractor.py
PaperMetadata(title, authors, year, bibtex_key, bibtex_entry, journal, volume, pages, doi, url, abstract, publisher, extraction_method)

# src/embeddings.py
EmbeddingResult(embedding: list[float], token_count: int, model: str)

# src/vector_store.py
ChunkRecord(id, paper_id, text, vector, bibtex_key, bibtex_entry, title, authors, year, journal, doi, url, pdf_path, pdf_hash, date_added, chunk_index, section_title, section_hierarchy, page_number, element_type, tags, notes, extraction_method)
```

## Common Development Tasks

### Adding a New MCP Tool

1. Add tool definition to `handle_list_tools()` in `src/mcp_server.py`
2. Create async handler function: `async def tool_name_tool(arguments: dict) -> list[types.TextContent]`
3. Add case to `handle_call_tool()` dispatcher
4. Tool should call `initialize_components()` to ensure all modules loaded

### Modifying the Database Schema

⚠️ **Breaking change** - requires database rebuild:
1. Update `_create_schema()` in `src/vector_store.py` (PyArrow schema)
2. Update `ChunkRecord` dataclass
3. Update `add_paper()` method to populate new fields
4. Users must delete `data/lancedb/` and re-run `scripts/initial_setup.py`

### Adding New Metadata Sources

1. Add new method to `MetadataExtractor` class (e.g., `_get_metadata_from_semanticscholar()`)
2. Add new `ExtractionMethod` enum value
3. Call new method in `extract_metadata()` strategy chain
4. Return `PaperMetadata` with appropriate `extraction_method`

### Cost Optimization

- Reduce `batch_size` in config if hitting OpenAI rate limits
- Adjust `max_chunk_tokens` to reduce chunk count (larger chunks = fewer embeddings)
- Cache embeddings for queries (not currently implemented)

## Logging

Logs written to `data/logs/`:
- `mcp_server.log` - MCP server operations
- `initial_setup.log` - Batch PDF processing (includes BibTeX file save status)
- Each module uses `setup_logger(__name__)` from `src/utils.py`

Log levels configured in `config.yaml` (`log_level: INFO`)

## Output Files

**Individual BibTeX Files** (`data/bibs/`):
- Automatically generated for each processed PDF
- Filename matches PDF filename (e.g., `paper.pdf` → `paper.bib`)
- Contains single BibTeX entry with metadata extracted via pdf2bib or fallback methods
- Useful for quick citation access without querying the database

## Error Handling Patterns

**Graceful degradation in metadata extraction**: Tries pdf2bib first (most reliable), then text-based DOI → CrossRef, then arXiv, then PDF metadata, then parsing. If pdf2bib is not installed, automatically falls back to text-based methods. Always returns some `PaperMetadata` (worst case: title from filename, Unknown authors).

**Retry logic**: Embeddings and API calls use exponential backoff (1s, 2s, 4s delays).

**Duplicate detection**: Papers checked by PDF hash before processing. Skipped silently with log entry.

**Corrupted PDFs**: Caught in `DocumentProcessor.process_pdf()`, logged, counted in failures.

## Important Implementation Notes

### LanceDB Quirks
- No easy in-place updates - store mutable fields (tags, notes) carefully
- `delete()` uses SQL-like WHERE syntax: `table.delete("bibtex_key = 'Smith2024'")`
- Search requires `.to_list()` to execute query (lazy evaluation)

### Docling Chunking
- Chunker uses tokenizer name (string), not actual tokenizer object
- `HybridChunker` balances semantic boundaries with token limits
- Chunk metadata depends on Docling's document structure - may be incomplete for poorly formatted PDFs

### BibTeX Key Collisions
- System auto-appends letters: `Smith2024` → `Smith2024a` → `Smith2024b`
- Pass `existing_keys` set to `generate_bibtex_key()` to avoid collisions
- Keys stored in both metadata and chunk records (denormalized)

### OpenAI Embedding Costs
- text-embedding-3-large: $0.00013 per 1K tokens
- Typical paper (20 pages): ~10-15 chunks, ~$0.015 total
- Cost tracked in `EmbeddingResult.token_count` and logged

### pdf2bib Integration
- **Primary extraction method** (Strategy 1) - extracts identifiers directly from PDF binary
- More reliable than text-based regex extraction - works even with scanned PDFs or complex layouts
- Supports DOI, arXiv, and PMID extraction in a single call
- Automatically queries appropriate APIs (CrossRef, arXiv) based on identifier type
- **Graceful fallback**: If not installed, system automatically uses text-based extraction methods
- Configure verbosity: `pdf2bib.config.set('verbose', True)` in `metadata_extractor.py` for debugging

## Development Workflow

1. Make code changes in `src/`
2. Test MCP tools manually or via Claude Desktop
3. For schema changes: delete `data/lancedb/` and reinitialize
4. Check logs in `data/logs/` for debugging
5. API keys in `.env` (never commit)

## Future Enhancements (Planned)

**Phase 2 (v1.5)**: `add_paper_from_url`, `verify_citation`, `update_bibtex`, `import_bibtex_file` tools

**Phase 3 (v2.0)**: Citation graph - extract references, build network, analyze co-citations

See `PRD.md` Section 6 for full roadmap.
