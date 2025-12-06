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
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
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

# Create directory structure
echo "Creating directory structure..."

# Notebooks directory
mkdir -p notebooks
echo "✓ Created notebooks/"

# Scripts directory
mkdir -p scripts
echo "✓ Created scripts/"

# Derivatives directories
mkdir -p derivatives/glm_tutorial
mkdir -p derivatives/rl_agent
mkdir -p derivatives/encoding
mkdir -p derivatives/presentation_cache
mkdir -p derivatives/figures
echo "✓ Created derivatives/ subdirectories"

echo ""

# Install Jupyter extensions
echo "Installing Jupyter extensions..."
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user 2>/dev/null || true
jupyter nbextension enable rise --py --sys-prefix
echo "✓ Jupyter extensions configured"

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
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Import Mario ROM (if using RL features):"
echo "   python3 -m retro.import /path/to/SuperMarioBros.nes"
echo "3. Ensure data is available in sourcedata/"
echo "   - sourcedata/mario/"
echo "   - sourcedata/mario.fmriprep/"
echo "   - sourcedata/mario.annotations/"
echo "4. Launch Jupyter: jupyter notebook"
echo "5. Start with notebooks/01_dataset_exploration.ipynb"
echo "   or notebooks/00_presentation_RISE.ipynb for the slideshow"
echo ""
echo "For RISE presentations:"
echo "  - Open 00_presentation_RISE.ipynb"
echo "  - Click the RISE button to start slideshow"
echo "  - Use spacebar to advance, shift+spacebar to go back"
echo ""
echo "Enjoy the tutorial!"
echo ""
