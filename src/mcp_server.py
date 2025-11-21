"""
MCP Server for Academic RAG Pipeline.
Provides tools for Claude Desktop to interact with the paper database.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import yaml

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp import types
import mcp.server.stdio

from src.document_processor import DocumentProcessor
from src.metadata_extractor import MetadataExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.bibliography import BibliographyManager, BibliographyEntry
from src.utils import setup_logger, compute_file_hash, save_bibtex_file, copy_pdf_to_database


# Set up logging
logger = setup_logger(__name__, log_file=Path("data/logs/mcp_server.log"))

# Initialize server
server = Server("academic-rag")

# Global instances (initialized on first use)
config = None
doc_processor = None
metadata_extractor = None
embedding_generator = None
vector_store = None
bibliography_manager = None


def load_config() -> dict:
    """Load configuration from config.yaml and environment variables."""
    global config

    if config is not None:
        return config

    # Determine project root (parent of src directory)
    project_root = Path(__file__).parent.parent.resolve()

    # Get config path from environment or use default
    config_path = os.getenv('CONFIG_PATH', 'config/config.yaml')
    config_file = Path(config_path)

    # If config path is relative, resolve it from project root
    if not config_file.is_absolute():
        config_file = project_root / config_file

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve relative paths in config to project root
    for path_key in ['lancedb_path', 'pdf_library_path', 'default_bib_output', 'pdfs_path']:
        if path_key in config and config[path_key]:
            path_value = Path(config[path_key])
            if not path_value.is_absolute():
                config[path_key] = str(project_root / path_value)

    # Override with environment variables
    config['openai_api_key'] = os.getenv('OPENAI_API_KEY', config.get('openai_api_key', ''))
    config['crossref_email'] = os.getenv('CROSSREF_EMAIL', config.get('crossref_email', ''))

    logger.info(f"Configuration loaded successfully from {config_file}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"LanceDB path: {config.get('lancedb_path')}")
    return config


def initialize_components():
    """Initialize all pipeline components."""
    global doc_processor, metadata_extractor, embedding_generator, vector_store, bibliography_manager

    cfg = load_config()

    # Initialize document processor
    if doc_processor is None:
        doc_processor = DocumentProcessor(
            max_chunk_tokens=cfg.get('max_chunk_tokens', 1000),
            chunk_overlap=cfg.get('chunk_overlap', 150),
            embedding_model=cfg.get('embedding_model', 'text-embedding-3-large')
        )

    # Initialize metadata extractor
    if metadata_extractor is None:
        metadata_extractor = MetadataExtractor(
            crossref_email=cfg.get('crossref_email')
        )

    # Initialize embedding generator
    if embedding_generator is None:
        api_key = cfg.get('openai_api_key')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")

        embedding_generator = EmbeddingGenerator(
            api_key=api_key,
            model=cfg.get('embedding_model', 'text-embedding-3-large'),
            dimensions=cfg.get('vector_dimension', 3072),
            batch_size=cfg.get('batch_size', 100)
        )

    # Initialize vector store
    if vector_store is None:
        db_path = Path(cfg.get('lancedb_path', 'data/lancedb'))
        vector_store = VectorStore(
            db_path=db_path,
            vector_dimension=cfg.get('vector_dimension', 3072)
        )
        vector_store.initialize_table()

    # Initialize bibliography manager
    if bibliography_manager is None:
        bibliography_manager = BibliographyManager()

    logger.info("All components initialized successfully")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name="search_papers",
            description="Search the paper database for relevant content using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-20, default: 5)",
                        "default": 5
                    },
                    "filter_section": {
                        "type": "string",
                        "description": "Filter by section type (Methods, Results, Discussion, Introduction). Optional."
                    },
                    "min_year": {
                        "type": "integer",
                        "description": "Only papers from this year onwards. Optional."
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="add_paper_from_file",
            description="Add a PDF paper to the database from a file path",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file"
                    },
                    "custom_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags",
                        "default": []
                    }
                },
                "required": ["file_path"]
            }
        ),
        types.Tool(
            name="generate_bibliography",
            description="Create a .bib file with specified papers",
            inputSchema={
                "type": "object",
                "properties": {
                    "bibtex_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of BibTeX keys to include"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Where to save the file",
                        "default": "./references.bib"
                    },
                    "include_abstracts": {
                        "type": "boolean",
                        "description": "Include abstract field",
                        "default": False
                    }
                },
                "required": ["bibtex_keys"]
            }
        ),
        types.Tool(
            name="get_paper_details",
            description="Retrieve complete information about a specific paper",
            inputSchema={
                "type": "object",
                "properties": {
                    "bibtex_key": {
                        "type": "string",
                        "description": "The BibTeX key of the paper"
                    }
                },
                "required": ["bibtex_key"]
            }
        ),
        types.Tool(
            name="database_stats",
            description="Get statistics about the paper database",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        types.Tool(
            name="list_recent_papers",
            description="Show recently added papers",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of papers to show",
                        "default": 10
                    }
                }
            }
        ),
        types.Tool(
            name="delete_paper",
            description="Delete a paper from the database and optionally remove associated files",
            inputSchema={
                "type": "object",
                "properties": {
                    "bibtex_key": {
                        "type": "string",
                        "description": "The BibTeX key of the paper to delete"
                    },
                    "delete_files": {
                        "type": "boolean",
                        "description": "Whether to delete associated PDF and .bib files (default: true)",
                        "default": True
                    }
                },
                "required": ["bibtex_key"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict
) -> list[types.TextContent]:
    """Handle tool execution."""
    try:
        initialize_components()

        if name == "search_papers":
            return await search_papers_tool(arguments)

        elif name == "add_paper_from_file":
            return await add_paper_from_file_tool(arguments)

        elif name == "generate_bibliography":
            return await generate_bibliography_tool(arguments)

        elif name == "get_paper_details":
            return await get_paper_details_tool(arguments)

        elif name == "database_stats":
            return await database_stats_tool(arguments)

        elif name == "list_recent_papers":
            return await list_recent_papers_tool(arguments)

        elif name == "delete_paper":
            return await delete_paper_tool(arguments)

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [types.TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


async def search_papers_tool(arguments: dict) -> list[types.TextContent]:
    """Search for papers in the database."""
    query = arguments.get("query")
    n_results = arguments.get("n_results", 5)
    filter_section = arguments.get("filter_section")
    min_year = arguments.get("min_year")

    logger.info(f"Searching for: {query}")

    # Generate query embedding
    query_embedding = embedding_generator.generate_embedding(query)

    # Search vector store
    results = vector_store.search(
        query_vector=query_embedding.embedding,
        n_results=n_results,
        filter_section=filter_section,
        min_year=min_year
    )

    # Format results
    if not results:
        return [types.TextContent(
            type="text",
            text="No results found for your query."
        )]

    output_lines = [f"Found {len(results)} relevant papers:\n"]

    for i, result in enumerate(results, 1):
        authors_list = result['authors'].split(',')
        authors_str = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors_str += " et al."

        output_lines.append(f"**[{result['bibtex_key']}]** {result['title']}")
        output_lines.append(f"📝 {authors_str} ({result['year']})")

        if result['journal']:
            output_lines.append(f"📍 {result['journal']}")

        if result['url']:
            output_lines.append(f"🔗 [Read online]({result['url']})")

        output_lines.append(
            f"\n**Relevant excerpt** (Section: {result['section_title']}, "
            f"Page {result['page_number'] or 'N/A'}):"
        )

        output_lines.append(result['text'])
        output_lines.append("\n---\n")

    return [types.TextContent(
        type="text",
        text="\n".join(output_lines)
    )]


async def add_paper_from_file_tool(arguments: dict) -> list[types.TextContent]:
    """Add a paper from a PDF file."""
    file_path = Path(arguments.get("file_path"))
    custom_tags = arguments.get("custom_tags", [])

    logger.info(f"Adding paper from: {file_path}")

    if not file_path.exists():
        return [types.TextContent(
            type="text",
            text=f"Error: File not found: {file_path}"
        )]

    # Check for duplicates
    pdf_hash = compute_file_hash(file_path)
    if vector_store.check_duplicate(pdf_hash):
        return [types.TextContent(
            type="text",
            text=f"Paper already exists in database (duplicate PDF detected): {file_path.name}"
        )]

    # Process PDF
    doc, chunks = doc_processor.process_pdf(file_path)

    # Extract text from first pages for metadata
    first_pages_text = doc_processor.extract_text_from_first_pages(file_path)

    # Extract metadata
    existing_keys = vector_store.get_all_bibtex_keys()
    metadata = metadata_extractor.extract_metadata(
        file_path,
        first_pages_text=first_pages_text,
        existing_keys=existing_keys
    )

    # Generate embeddings
    chunk_texts = [chunk.text for chunk in chunks]
    embedding_results = embedding_generator.generate_embeddings_batch(chunk_texts)
    embeddings = [result.embedding for result in embedding_results]

    # Copy PDF to database storage
    pdfs_dir = Path(config.get('pdfs_path', 'data/pdfs'))
    try:
        copied_pdf_path = copy_pdf_to_database(
            source_pdf=file_path,
            bibtex_key=metadata.bibtex_key,
            output_dir=pdfs_dir
        )
        logger.info(f"Copied PDF to database: {copied_pdf_path}")
        pdf_copied = True
    except Exception as e:
        logger.warning(f"Failed to copy PDF: {e}")
        copied_pdf_path = file_path
        pdf_copied = False

    # Add to vector store (use copied path if available)
    num_chunks = vector_store.add_paper(
        metadata=metadata,
        chunks=chunks,
        embeddings=embeddings,
        pdf_path=copied_pdf_path,
        pdf_hash=pdf_hash,
        tags=custom_tags
    )

    # Save individual BibTeX file
    bibs_dir = Path(config.get('bibs_output_path', 'data/bibs'))
    try:
        bib_file_path = save_bibtex_file(
            bibtex_entry=metadata.bibtex_entry,
            bibtex_key=metadata.bibtex_key,
            output_dir=bibs_dir
        )
        logger.info(f"Saved BibTeX file: {bib_file_path}")
        bib_saved = True
    except Exception as e:
        logger.warning(f"Failed to save BibTeX file: {e}")
        bib_saved = False

    # Get embedding stats
    stats = embedding_generator.get_embedding_stats(embedding_results)

    output = f"""
Successfully added paper to database!

**Title:** {metadata.title}
**Authors:** {', '.join(metadata.authors)}
**Year:** {metadata.year}
**BibTeX Key:** {metadata.bibtex_key}
**DOI:** {metadata.doi or 'N/A'}
**URL:** {metadata.url or 'N/A'}
**Extraction Method:** {metadata.extraction_method.value}

**Indexed:** {num_chunks} chunks
**Tokens Processed:** {stats['total_tokens']}
**Estimated Cost:** ${stats['estimated_cost_usd']:.4f}
{'**PDF File:** Copied to ' + str(pdfs_dir / (metadata.bibtex_key + '.pdf')) if pdf_copied else '**PDF File:** Failed to copy (using original path)'}
{'**BibTeX File:** Saved to ' + str(bibs_dir / (metadata.bibtex_key + '.bib')) if bib_saved else '**BibTeX File:** Failed to save'}

The paper is now searchable in your database.
"""

    return [types.TextContent(type="text", text=output)]


async def generate_bibliography_tool(arguments: dict) -> list[types.TextContent]:
    """Generate a bibliography file."""
    bibtex_keys = arguments.get("bibtex_keys", [])
    output_path = Path(arguments.get("output_path", "./references.bib"))
    include_abstracts = arguments.get("include_abstracts", False)

    logger.info(f"Generating bibliography with {len(bibtex_keys)} entries")

    # Retrieve papers from database
    entries = []
    missing_keys = []

    for key in bibtex_keys:
        paper = vector_store.get_paper_by_key(key)
        if paper:
            entry = BibliographyEntry(
                bibtex_key=paper['bibtex_key'],
                bibtex_entry=paper['bibtex_entry'],
                title=paper['title'],
                authors=paper['authors'],
                year=paper['year']
            )
            entries.append(entry)
        else:
            missing_keys.append(key)

    # Generate bibliography file
    result = bibliography_manager.generate_bibliography_file(
        entries=entries,
        output_path=output_path,
        include_abstracts=include_abstracts
    )

    output_lines = [
        f"Bibliography generated: {output_path}",
        f"\nIncluded: {result['success_count']} entries"
    ]

    if missing_keys:
        output_lines.append(f"\nMissing keys: {', '.join(missing_keys)}")

    if result['errors']:
        output_lines.append(f"\nErrors: {', '.join(result['errors'])}")

    return [types.TextContent(
        type="text",
        text="\n".join(output_lines)
    )]


async def get_paper_details_tool(arguments: dict) -> list[types.TextContent]:
    """Get details about a specific paper."""
    bibtex_key = arguments.get("bibtex_key")

    logger.info(f"Getting details for: {bibtex_key}")

    paper = vector_store.get_paper_by_key(bibtex_key)

    if not paper:
        return [types.TextContent(
            type="text",
            text=f"Paper not found: {bibtex_key}"
        )]

    # Get chunk count
    paper_id = paper['pdf_hash'][:16]
    chunks = vector_store.get_paper_chunks(paper_id)

    output = f"""
**Paper Details**

**Title:** {paper['title']}
**Authors:** {', '.join(paper['authors'])}
**Year:** {paper['year']}
**BibTeX Key:** {paper['bibtex_key']}

**Publication Info:**
- Journal: {paper['journal'] or 'N/A'}
- DOI: {paper['doi'] or 'N/A'}
- URL: {paper['url'] or 'N/A'}

**Database Info:**
- Chunks Indexed: {len(chunks)}
- Date Added: {paper['date_added']}
- Extraction Method: {paper['extraction_method']}
- PDF Path: {paper['pdf_path']}

**BibTeX Entry:**
```bibtex
{paper['bibtex_entry']}
```
"""

    return [types.TextContent(type="text", text=output)]


async def database_stats_tool(arguments: dict) -> list[types.TextContent]:
    """Get database statistics."""
    logger.info("Getting database statistics")

    stats = vector_store.get_statistics()

    output = f"""
**Database Statistics**

**Papers:** {stats['total_papers']}
**Chunks:** {stats['total_chunks']}
**Average Publication Year:** {stats['average_year']}

**Papers by Year:**
"""

    # Sort years
    sorted_years = sorted(stats['year_distribution'].items(), reverse=True)
    for year, count in sorted_years[:10]:  # Show top 10 years
        output += f"\n  {year}: {count} papers"

    if len(sorted_years) > 10:
        output += f"\n  ... and {len(sorted_years) - 10} more years"

    output += f"\n\n**Database Path:** {stats['database_path']}"

    return [types.TextContent(type="text", text=output)]


async def list_recent_papers_tool(arguments: dict) -> list[types.TextContent]:
    """List recently added papers."""
    n = arguments.get("n", 10)

    logger.info(f"Listing {n} recent papers")

    papers = vector_store.list_recent_papers(n)

    if not papers:
        return [types.TextContent(
            type="text",
            text="No papers in database."
        )]

    output_lines = [f"**Recently Added Papers** (showing {len(papers)}):\n"]

    for i, paper in enumerate(papers, 1):
        authors_str = ", ".join(paper['authors'][:2])
        if len(paper['authors']) > 2:
            authors_str += " et al."

        output_lines.append(
            f"{i}. **[{paper['bibtex_key']}]** {paper['title']}"
        )
        output_lines.append(f"   {authors_str} ({paper['year']})")
        output_lines.append(f"   Added: {paper['date_added'][:10]}\n")

    return [types.TextContent(
        type="text",
        text="\n".join(output_lines)
    )]


async def delete_paper_tool(arguments: dict) -> list[types.TextContent]:
    """Delete a paper from the database and optionally delete associated files."""
    bibtex_key = arguments.get("bibtex_key")
    delete_files = arguments.get("delete_files", True)

    logger.info(f"Deleting paper: {bibtex_key} (delete_files={delete_files})")

    # First check if paper exists and get its details
    paper = vector_store.get_paper_by_key(bibtex_key)

    if not paper:
        return [types.TextContent(
            type="text",
            text=f"Error: Paper not found in database: {bibtex_key}"
        )]

    # Delete from database
    try:
        deleted_count = vector_store.delete_paper(bibtex_key)
        logger.info(f"Deleted {deleted_count} chunks for paper {bibtex_key}")
    except Exception as e:
        logger.error(f"Error deleting paper from database: {e}", exc_info=True)
        return [types.TextContent(
            type="text",
            text=f"Error deleting paper from database: {str(e)}"
        )]

    output_lines = [
        f"Successfully deleted paper from database: {bibtex_key}",
        f"- Title: {paper['title']}",
        f"- Authors: {', '.join(paper['authors'])}",
        f"- Deleted {deleted_count} chunks"
    ]

    # Delete associated files if requested
    if delete_files:
        files_deleted = []
        files_not_found = []

        # Delete PDF file
        pdfs_dir = Path(config.get('pdfs_path', 'data/pdfs'))
        pdf_path = pdfs_dir / f"{bibtex_key}.pdf"

        if pdf_path.exists():
            try:
                pdf_path.unlink()
                files_deleted.append(f"PDF: {pdf_path}")
                logger.info(f"Deleted PDF file: {pdf_path}")
            except Exception as e:
                logger.error(f"Error deleting PDF file: {e}", exc_info=True)
                output_lines.append(f"- Warning: Failed to delete PDF file: {e}")
        else:
            files_not_found.append(f"PDF: {pdf_path}")
            logger.info(f"PDF file not found (may have been manually deleted): {pdf_path}")

        # Delete BibTeX file
        bibs_dir = Path(config.get('bibs_output_path', 'data/bibs'))
        bib_path = bibs_dir / f"{bibtex_key}.bib"

        if bib_path.exists():
            try:
                bib_path.unlink()
                files_deleted.append(f".bib: {bib_path}")
                logger.info(f"Deleted BibTeX file: {bib_path}")
            except Exception as e:
                logger.error(f"Error deleting BibTeX file: {e}", exc_info=True)
                output_lines.append(f"- Warning: Failed to delete BibTeX file: {e}")
        else:
            files_not_found.append(f".bib: {bib_path}")
            logger.info(f"BibTeX file not found (may have been manually deleted): {bib_path}")

        # Add file deletion summary to output
        if files_deleted:
            output_lines.append(f"\nDeleted files:")
            for file_desc in files_deleted:
                output_lines.append(f"  - {file_desc}")

        if files_not_found:
            output_lines.append(f"\nFiles not found (already deleted or never created):")
            for file_desc in files_not_found:
                output_lines.append(f"  - {file_desc}")
    else:
        output_lines.append("\n- Associated files were not deleted (delete_files=False)")

    return [types.TextContent(
        type="text",
        text="\n".join(output_lines)
    )]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="academic-rag",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
