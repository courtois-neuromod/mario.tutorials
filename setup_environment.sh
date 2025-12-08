#!/bin/bash
# Setup script for Mario fMRI Tutorial
# This script sets up the Python environment and directory structure

set -e  # Exit on error

echo "================================================"
echo "Mario fMRI Tutorial - Environment Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Recommend Python 3.9+
required_version="3.9"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "WARNING: Python 3.9+ is recommended. You have $python_version"
fi

echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "env" ]; then
    python3 -m venv env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate
echo "✓ Virtual environment activated"

echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"

echo ""

# Install requirements
echo "Installing Python packages (this may take a few minutes)..."
pip install -r requirements.txt
echo "✓ All packages installed"

echo ""

# Install Jupyter Notebook extensions
echo "Installing Jupyter Notebook extensions..."
# Note: RISE is the slideshow extension for classic Jupyter Notebook
# It installs as a standard pip package with notebook 6.x
echo "✓ Jupyter Notebook extensions configured (RISE)"

echo ""

# Check for stable-retro ROM
echo "Checking gym-retro setup..."
if python3 -c "import retro" 2>/dev/null; then
    echo "✓ stable-retro is installed"
    echo ""
    echo "NOTE: You need to import the Super Mario Bros ROM"
    echo "Run: python3 -m retro.import /path/to/SuperMarioBros.nes"
    echo "The ROM should be located in sourcedata/mario.stimuli/SuperMarioBros-Nes/"
else
    echo "WARNING: stable-retro not found or failed to import"
fi

echo ""

# Summary
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source env/bin/activate"
echo "2. Import Mario ROM (if using RL features):"
echo "   python3 -m retro.import /path/to/SuperMarioBros.nes"
echo "3. Ensure data is available in sourcedata/"
echo "   - sourcedata/mario/"
echo "   - sourcedata/mario.fmriprep/"
echo "   - sourcedata/mario.annotations/"
echo "4. Launch Jupyter Notebook: jupyter notebook"
echo "5. Start with any tutorial notebook in the notebooks/ folder"
echo ""
echo "For RISE slideshow presentations:"
echo "  - Open any tutorial notebook (e.g., 01_event_based_analysis.ipynb)"
echo "  - Click the bar chart icon in the toolbar to start slideshow"
echo "  - Or use View > Cell Toolbar > Slideshow to configure slides"
echo "  - Use spacebar to advance, shift+spacebar to go back"
echo "  - Press Esc to exit slideshow mode"
echo ""
echo "Enjoy the tutorial!"
echo ""
