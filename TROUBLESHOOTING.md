# Troubleshooting Guide

## Common Issues and Solutions

### Multiprocessing Resource Tracker Warning

**Issue**: You see a warning like this when running `initial_setup.py`:
```
/Users/joe/.pyenv/versions/3.12.12/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

**Cause**: Docling's `DocumentConverter` uses multiprocessing internally for PDF processing. When the script exits, Python's resource tracker detects that semaphore objects weren't explicitly cleaned up.

**Impact**: This is a **non-critical warning**. Python will clean up these resources automatically. It doesn't affect functionality or data integrity.

**Solution**: The codebase has been updated with two mitigations:

1. **Added cleanup handler** in `src/document_processor.py`:
   - The `DocumentProcessor` class now has a `__del__()` method that explicitly cleans up the Docling converter

2. **Added warning suppression** in `scripts/initial_setup.py`:
   - The script now filters out resource tracker warnings since they're expected from Docling's multiprocessing

3. **Added explicit cleanup** in `scripts/initial_setup.py`:
   - The main function now uses a try-finally block to explicitly delete components

These changes ensure cleaner shutdown and prevent the warning from appearing.

### Testing the Fix

If you want to verify the fix works, the warning should no longer appear when running:
```bash
source .venv/bin/activate
python scripts/initial_setup.py
```

---

## Other Common Issues

### OpenAI API Key Not Found
**Solution**: Ensure `OPENAI_API_KEY` is set in your `.env` file:
```bash
OPENAI_API_KEY=sk-your-key-here
```

### PDF Library Path Not Found
**Solution**: Check that `pdf_library_path` in `config/config.yaml` points to a valid directory containing PDFs.

### LanceDB Permission Errors
**Solution**: Ensure the `data/lancedb/` directory has write permissions:
```bash
chmod -R u+w data/lancedb/
```
