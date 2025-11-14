"""
Utility functions for the Academic RAG Pipeline.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        name: Name of the logger
        log_file: Optional path to log file
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if log file specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file for duplicate detection.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract DOI from text using regex patterns.

    NOTE: This is a fallback method. The primary DOI extraction is handled by
    pdf2bib in metadata_extractor.py, which extracts identifiers directly from
    PDF binary content and is more reliable. This text-based extraction is used
    only when pdf2bib is not available or fails.

    Args:
        text: Text to search for DOI

    Returns:
        DOI string if found, None otherwise
    """
    # Common DOI patterns
    patterns = [
        r'10\.\d{4,}/[^\s]+',  # Standard DOI format
        r'doi:\s*10\.\d{4,}/[^\s]+',  # With 'doi:' prefix
        r'DOI:\s*10\.\d{4,}/[^\s]+',  # With 'DOI:' prefix
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            doi = match.group(0)
            # Remove 'doi:' or 'DOI:' prefix if present
            doi = re.sub(r'^(doi|DOI):\s*', '', doi)
            # Clean up trailing punctuation
            doi = doi.rstrip('.,;:')
            return doi

    return None


def extract_arxiv_id_from_text(text: str) -> Optional[str]:
    """
    Extract arXiv ID from text.

    NOTE: This is a fallback method. The primary arXiv ID extraction is handled by
    pdf2bib in metadata_extractor.py. This text-based extraction is used only when
    pdf2bib is not available or fails.

    Args:
        text: Text to search for arXiv ID

    Returns:
        arXiv ID if found, None otherwise
    """
    # arXiv ID patterns
    patterns = [
        r'arXiv:\s*(\d{4}\.\d{4,5})',  # New format: arXiv:YYMM.NNNNN
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',  # URL format
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def extract_pubmed_id_from_text(text: str) -> Optional[str]:
    """
    Extract PubMed ID from text.

    NOTE: This is a fallback method. The primary PMID extraction is handled by
    pdf2bib in metadata_extractor.py. This text-based extraction is used only when
    pdf2bib is not available or fails.

    Args:
        text: Text to search for PMID

    Returns:
        PMID if found, None otherwise
    """
    # PubMed ID patterns
    patterns = [
        r'PMID:\s*(\d{7,8})',  # PMID: format
        r'pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,8})',  # URL format
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def generate_bibtex_key(authors: list[str], year: int, existing_keys: set[str]) -> str:
    """
    Generate a BibTeX citation key from authors and year.

    Args:
        authors: List of author names
        year: Publication year
        existing_keys: Set of already used keys for collision detection

    Returns:
        Unique BibTeX key (e.g., 'Smith2024', 'Smith2024a', etc.)
    """
    if not authors:
        base_key = f"Unknown{year}"
    else:
        # Extract first author's last name
        first_author = authors[0]
        # Handle "Last, First" or "First Last" formats
        if ',' in first_author:
            last_name = first_author.split(',')[0].strip()
        else:
            parts = first_author.split()
            last_name = parts[-1] if parts else "Unknown"

        # Clean the last name (remove non-alphanumeric characters)
        last_name = re.sub(r'[^a-zA-Z]', '', last_name)
        base_key = f"{last_name}{year}"

    # Handle collisions by appending letters
    key = base_key
    suffix = ord('a')
    while key in existing_keys:
        key = f"{base_key}{chr(suffix)}"
        suffix += 1
        if suffix > ord('z'):
            # If we run out of letters, start using numbers
            key = f"{base_key}_{len(existing_keys)}"
            break

    return key


def clean_text_for_embedding(text: str) -> str:
    """
    Clean text before generating embeddings.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def format_authors_for_bibtex(authors: list[str]) -> str:
    """
    Format author list for BibTeX entry.

    Args:
        authors: List of author names

    Returns:
        BibTeX-formatted author string (e.g., "Last1, First1 and Last2, First2")
    """
    if not authors:
        return "Unknown"

    # Join with 'and' for BibTeX format
    return " and ".join(authors)


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO formatted timestamp string
    """
    return datetime.now().isoformat()


def validate_bibtex_entry(entry: str) -> bool:
    """
    Validate that a BibTeX entry has proper format.

    Args:
        entry: BibTeX entry string

    Returns:
        True if valid, False otherwise
    """
    # Basic validation: check for @type{key, structure
    if not re.match(r'@\w+\{[\w-]+,', entry):
        return False

    # Check for required fields (at least title and year)
    has_title = bool(re.search(r'title\s*=\s*\{[^}]+\}', entry, re.IGNORECASE))
    has_year = bool(re.search(r'year\s*=\s*\{?\d{4}\}?', entry, re.IGNORECASE))

    return has_title and has_year


def find_pdf_files(directory: Path, recursive: bool = True) -> list[Path]:
    """
    Find all PDF files in a directory.

    Args:
        directory: Directory to search
        recursive: Whether to search recursively

    Returns:
        List of PDF file paths
    """
    if recursive:
        pdf_files = list(directory.rglob("*.pdf"))
    else:
        pdf_files = list(directory.glob("*.pdf"))

    return sorted(pdf_files)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


def save_bibtex_file(bibtex_entry: str, bibtex_key: str, output_dir: Path, pdf_filename: Optional[str] = None) -> Path:
    """
    Save a BibTeX entry to an individual .bib file.

    Args:
        bibtex_entry: The BibTeX entry string to save
        bibtex_key: The BibTeX citation key (e.g., "Smith2024")
        output_dir: Directory where .bib files should be saved
        pdf_filename: Optional PDF filename to use as base (without extension)

    Returns:
        Path to the saved .bib file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename: use PDF filename if provided, otherwise use bibtex_key
    if pdf_filename:
        # Remove .pdf extension if present
        base_name = pdf_filename.replace('.pdf', '').replace('.PDF', '')
    else:
        base_name = bibtex_key

    # Sanitize the filename
    base_name = sanitize_filename(base_name)

    # Create .bib file path
    bib_path = output_dir / f"{base_name}.bib"

    # Write BibTeX entry
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(bibtex_entry)
        f.write('\n')  # Add trailing newline

    return bib_path
