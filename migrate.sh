#!/bin/bash

echo "Starting migration preparation..."

# Clean up unnecessary files
echo "Cleaning up unnecessary files..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name ".DS_Store" -delete
find . -type d -name ".pytest_cache" -exec rm -r {} +
find . -type d -name ".ruff_cache" -exec rm -r {} +

# Create a requirements file with exact versions
echo "Creating frozen requirements file..."
pip freeze > requirements.frozen.txt


echo "Migration preparation complete!"
echo "The following actions were performed:"
echo "1. Removed all __pycache__ directories"
echo "2. Removed all .DS_Store files"
echo "3. Removed all .pytest_cache directories"
echo "4. Removed all .ruff_cache directories"
echo "5. Created requirements.frozen.txt with exact package versions"
