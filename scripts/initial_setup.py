#!/usr/bin/env python3
"""
Initial setup script for Academic RAG Pipeline.
Processes all PDFs in the configured library directory and creates the vector database.
"""

import os
import sys
from pathlib import Path
import yaml
import logging
import warnings
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Suppress resource tracker warnings from multiprocessing (used internally by Docling)
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processor import DocumentProcessor
from src.metadata_extractor import MetadataExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.utils import setup_logger, find_pdf_files, compute_file_hash, save_bibtex_file, copy_pdf_to_database


console = Console()


def load_configuration():
    """Load configuration from config file and environment."""
    console.print("[bold blue]Loading configuration...[/bold blue]")

    # Load environment variables
    load_dotenv()

    # Load config file
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        console.print("[bold red]Error: config/config.yaml not found![/bold red]")
        console.print("Please create the configuration file first.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    config['openai_api_key'] = os.getenv('OPENAI_API_KEY', config.get('openai_api_key', ''))
    config['crossref_email'] = os.getenv('CROSSREF_EMAIL', config.get('crossref_email', ''))

    # Validate required fields
    if not config.get('openai_api_key'):
        console.print("[bold red]Error: OPENAI_API_KEY not found![/bold red]")
        console.print("Set it in .env file or config.yaml")
        sys.exit(1)

    if not config.get('pdf_library_path'):
        console.print("[bold red]Error: pdf_library_path not set in config.yaml![/bold red]")
        sys.exit(1)

    console.print("[green]✓[/green] Configuration loaded")
    return config


def initialize_components(config):
    """Initialize all pipeline components."""
    console.print("[bold blue]Initializing components...[/bold blue]")

    # Set up logging
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger('initial_setup', log_file=log_dir / 'initial_setup.log')

    # Initialize document processor
    doc_processor = DocumentProcessor(
        max_chunk_tokens=config.get('max_chunk_tokens', 1000),
        chunk_overlap=config.get('chunk_overlap', 150),
        embedding_model=config.get('embedding_model', 'text-embedding-3-large')
    )
    console.print("[green]✓[/green] Document processor initialized")

    # Initialize metadata extractor
    metadata_extractor = MetadataExtractor(
        crossref_email=config.get('crossref_email')
    )
    console.print("[green]✓[/green] Metadata extractor initialized")

    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        api_key=config['openai_api_key'],
        model=config.get('embedding_model', 'text-embedding-3-large'),
        dimensions=config.get('vector_dimension', 3072),
        batch_size=config.get('batch_size', 100)
    )
    console.print("[green]✓[/green] Embedding generator initialized")

    # Initialize vector store
    db_path = Path(config.get('lancedb_path', 'data/lancedb'))
    vector_store = VectorStore(
        db_path=db_path,
        vector_dimension=config.get('vector_dimension', 3072)
    )
    vector_store.initialize_table()
    console.print("[green]✓[/green] Vector store initialized")

    return doc_processor, metadata_extractor, embedding_generator, vector_store, logger


def process_pdf_library(config, doc_processor, metadata_extractor, embedding_generator, vector_store, logger):
    """Process all PDFs in the library."""
    pdf_library_path = Path(config['pdf_library_path'])

    if not pdf_library_path.exists():
        console.print(f"[bold red]Error: PDF library path not found: {pdf_library_path}[/bold red]")
        sys.exit(1)

    # Set up bibs directory
    bibs_dir = Path(config.get('bibs_output_path', 'data/bibs'))
    bibs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"BibTeX files will be saved to: {bibs_dir}")

    # Set up pdfs directory
    pdfs_dir = Path(config.get('pdfs_path', 'data/pdfs'))
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"PDF files will be copied to: {pdfs_dir}")

    # Find all PDFs
    console.print(f"\n[bold blue]Scanning for PDFs in: {pdf_library_path}[/bold blue]")
    pdf_files = find_pdf_files(pdf_library_path, recursive=True)

    if not pdf_files:
        console.print("[yellow]No PDF files found in the library directory.[/yellow]")
        return

    console.print(f"[green]Found {len(pdf_files)} PDF files[/green]\n")

    # Get existing keys to avoid collisions
    existing_keys = vector_store.get_all_bibtex_keys()

    # Process each PDF
    processed = 0
    skipped = 0
    failed = 0
    total_cost = 0.0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Processing PDFs...", total=len(pdf_files))

        for pdf_path in pdf_files:
            progress.update(task, description=f"[cyan]Processing: {pdf_path.name}")

            try:
                # Check for duplicates
                pdf_hash = compute_file_hash(pdf_path)
                if vector_store.check_duplicate(pdf_hash):
                    logger.info(f"Skipping duplicate: {pdf_path.name}")
                    skipped += 1
                    progress.advance(task)
                    continue

                # Process PDF
                doc, chunks = doc_processor.process_pdf(pdf_path)

                # Extract metadata
                first_pages_text = doc_processor.extract_text_from_first_pages(pdf_path)
                metadata = metadata_extractor.extract_metadata(
                    pdf_path,
                    first_pages_text=first_pages_text,
                    existing_keys=existing_keys
                )

                # Add key to existing set
                existing_keys.add(metadata.bibtex_key)

                # Generate embeddings
                chunk_texts = [chunk.text for chunk in chunks]
                embedding_results = embedding_generator.generate_embeddings_batch(chunk_texts)
                embeddings = [result.embedding for result in embedding_results]

                # Track cost
                stats = embedding_generator.get_embedding_stats(embedding_results)
                total_cost += stats['estimated_cost_usd']

                # Copy PDF to database storage
                try:
                    copied_pdf_path = copy_pdf_to_database(
                        source_pdf=pdf_path,
                        bibtex_key=metadata.bibtex_key,
                        output_dir=pdfs_dir
                    )
                    logger.info(f"Copied PDF to database: {copied_pdf_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to copy PDF for {pdf_path.name}: {e}")
                    copied_pdf_path = pdf_path

                # Add to vector store (use copied path)
                vector_store.add_paper(
                    metadata=metadata,
                    chunks=chunks,
                    embeddings=embeddings,
                    pdf_path=copied_pdf_path,
                    pdf_hash=pdf_hash
                )

                # Save individual BibTeX file
                try:
                    bib_file_path = save_bibtex_file(
                        bibtex_entry=metadata.bibtex_entry,
                        bibtex_key=metadata.bibtex_key,
                        output_dir=bibs_dir
                    )
                    logger.info(f"Saved BibTeX file: {bib_file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to save BibTeX file for {pdf_path.name}: {e}")

                processed += 1
                logger.info(f"Successfully processed: {pdf_path.name} ({metadata.bibtex_key})")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
                failed += 1

            progress.advance(task)

    # Print summary
    console.print("\n[bold green]Processing Complete![/bold green]\n")
    console.print(f"[green]✓[/green] Successfully processed: {processed} papers")
    if skipped > 0:
        console.print(f"[yellow]⊘[/yellow] Skipped (duplicates): {skipped} papers")
    if failed > 0:
        console.print(f"[red]✗[/red] Failed: {failed} papers")

    console.print(f"\n[bold]Total estimated cost:[/bold] ${total_cost:.4f}")

    # Show database stats
    stats = vector_store.get_statistics()
    console.print(f"\n[bold]Database Statistics:[/bold]")
    console.print(f"  Total papers: {stats['total_papers']}")
    console.print(f"  Total chunks: {stats['total_chunks']}")
    console.print(f"  Database path: {stats['database_path']}")


def main():
    """Main setup function."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  Academic RAG Pipeline - Initial Setup[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════[/bold cyan]\n")

    # Load configuration
    config = load_configuration()

    # Initialize components
    doc_processor, metadata_extractor, embedding_generator, vector_store, logger = initialize_components(config)

    try:
        # Process PDF library
        process_pdf_library(config, doc_processor, metadata_extractor, embedding_generator, vector_store, logger)

        console.print("\n[bold green]Setup complete! You can now use the MCP server with Claude Desktop.[/bold green]\n")
    finally:
        # Explicit cleanup to prevent resource leaks
        logger.info("Cleaning up resources...")
        del doc_processor
        del metadata_extractor
        del embedding_generator
        del vector_store


if __name__ == "__main__":
    main()
