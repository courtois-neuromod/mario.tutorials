#!/bin/bash
# ==============================================================================
# Mario fMRI Tutorial - Data Download Script
# ==============================================================================
# This script downloads the CNeuromod Mario dataset for the tutorial
# Focus: Single subject (sub-01), single session (ses-010)
# Total download: ~7-8 GB
# Duration: ~20-30 minutes (depending on connection)
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SUBJECT="sub-01"
SESSION="ses-010"
SOURCEDATA_DIR="sourcedata"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ==============================================================================
# Main Installation
# ==============================================================================

print_header "Mario fMRI Tutorial - Data Download"

echo "This script will download:"
echo "  • Subject: ${SUBJECT}"
echo "  • Session: ${SESSION}"
echo "  • Total size: ~7-8 GB"
echo "  • Location: $(pwd)/${SOURCEDATA_DIR}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Installation cancelled"
    exit 0
fi

# ------------------------------------------------------------------------------
# Install Dependencies
# ------------------------------------------------------------------------------

print_header "Step 1/6: Installing Dependencies"

if ! command -v datalad &> /dev/null; then
    print_info "Installing datalad..."
    pip install datalad invoke airoh
    print_success "Dependencies installed"
else
    print_success "datalad already installed"
fi

# ------------------------------------------------------------------------------
# Create Directory Structure
# ------------------------------------------------------------------------------

print_header "Step 2/6: Creating Directory Structure"

if [ ! -d "${SOURCEDATA_DIR}" ]; then
    mkdir -p "${SOURCEDATA_DIR}"
    print_success "Created ${SOURCEDATA_DIR}/"
else
    print_warning "${SOURCEDATA_DIR}/ already exists"
fi

cd "${SOURCEDATA_DIR}"

# ------------------------------------------------------------------------------
# Download Dataset Repositories
# ------------------------------------------------------------------------------

print_header "Step 3/6: Installing Dataset Repositories"

# cneuromod.processed (anatomical data)
if [ ! -d "cneuromod.processed" ]; then
    print_info "Installing cneuromod.processed..."
    datalad install git@github.com:courtois-neuromod/cneuromod.processed
    print_success "cneuromod.processed installed"
else
    print_warning "cneuromod.processed already exists"
fi

# mario (raw BIDS data)
if [ ! -d "mario" ]; then
    print_info "Installing mario..."
    datalad install git@github.com:courtois-neuromod/mario
    print_success "mario installed"
else
    print_warning "mario already exists"
fi

# mario.stimuli (game ROM and integration files)
if [ ! -d "mario/mario.stimuli" ]; then
    # Check if AWS credentials are available
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        print_warning "mario.stimuli requires AWS credentials - skipping download"
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}MARIO.STIMULI - AWS CREDENTIALS REQUIRED${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "The mario.stimuli dataset contains the Super Mario Bros 3 ROM and"
        echo "integration files. This data is stored on AWS S3 and requires"
        echo "access credentials."
        echo ""
        echo "Options to obtain access:"
        echo ""
        echo "  1. Request AWS credentials from the CNeuromod team:"
        echo "     • Contact: courtois-neuromod-admin@criugm.qc.ca"
        echo "     • Specify you need access to mario.stimuli dataset"
        echo ""
        echo "  2. Obtain the ROM legally:"
        echo "     • Purchase Super Mario Bros 3 for NES"
        echo "     • Extract ROM using legal methods"
        echo "     • Place in sourcedata/mario/mario.stimuli/"
        echo ""
        echo "Note: The tutorial can proceed without mario.stimuli, but some"
        echo "      visualizations and replay analyses will be limited."
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
    else
        print_info "Installing mario.stimuli..."
        cd mario
        datalad install git@github.com:courtois-neuromod/mario.stimuli
        cd mario.stimuli
        print_info "Downloading stimuli files..."
        datalad get .
        cd ../..
        print_success "mario.stimuli downloaded"
    fi
else
    print_warning "mario.stimuli already exists"
fi

# mario.fmriprep (preprocessed fMRI)
if [ ! -d "mario.fmriprep" ]; then
    print_info "Installing mario.fmriprep..."
    datalad install git@github.com:courtois-neuromod/mario.fmriprep
    print_success "mario.fmriprep installed"
else
    print_warning "mario.fmriprep already exists"
fi

# mario.annotations (behavioral events)
if [ ! -d "mario.annotations" ]; then
    print_info "Installing mario.annotations..."
    git clone git@github.com:courtois-neuromod/mario.annotations
    print_success "mario.annotations cloned"
else
    print_warning "mario.annotations already exists"
fi

# mario.scenes (scene segmentation)
if [ ! -d "mario.scenes" ]; then
    print_info "Installing mario.scenes..."
    git clone git@github.com:courtois-neuromod/mario.scenes
    print_success "mario.scenes cloned"
else
    print_warning "mario.scenes already exists"
fi

# ------------------------------------------------------------------------------
# Download Session Data
# ------------------------------------------------------------------------------

print_header "Step 4/6: Downloading Session Data (${SUBJECT}/${SESSION})"

print_info "This will download ~7-8 GB of data..."
echo ""

# Raw BIDS data (events and gamelogs)
print_info "Downloading raw BIDS data..."
datalad get "mario/${SUBJECT}/${SESSION}/func"/*.tsv 2>/dev/null || print_warning "Some TSV files may not exist"
datalad get "mario/${SUBJECT}/${SESSION}/gamelogs"/*.bk2 2>/dev/null || print_warning "Some BK2 files may not exist"
print_success "Raw BIDS data downloaded"

# Preprocessed fMRI data
print_info "Downloading preprocessed fMRI data (this may take 10-15 minutes)..."
datalad get "mario.fmriprep/${SUBJECT}/${SESSION}/func"/*space-MNI152NLin2009cAsym* 2>/dev/null || print_warning "Some fMRIPrep files may not exist"
datalad get "mario.fmriprep/${SUBJECT}/${SESSION}/func"/*desc-confounds* 2>/dev/null || print_warning "Some confounds may not exist"
print_success "Preprocessed fMRI data downloaded"

# Anatomical data
print_info "Downloading anatomical data..."
datalad get "cneuromod.processed/smriprep/${SUBJECT}/anat"/*space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz 2>/dev/null || print_warning "Anatomical may not exist"
datalad get "cneuromod.processed/smriprep/${SUBJECT}/anat"/*space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz 2>/dev/null || print_warning "Brain mask may not exist"
print_success "Anatomical data downloaded"

# ------------------------------------------------------------------------------
# Verify Downloads
# ------------------------------------------------------------------------------

print_header "Step 5/6: Verifying Downloads"

# Check for key files
errors=0

# Raw data
if [ ! -f "mario/${SUBJECT}/${SESSION}/func/${SUBJECT}_${SESSION}_task-mario_run-01_events.tsv" ]; then
    print_error "Raw events file not found"
    ((errors++))
else
    print_success "Raw events verified"
fi

# Preprocessed BOLD
bold_files=$(find "mario.fmriprep/${SUBJECT}/${SESSION}/func" -name "*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" 2>/dev/null | wc -l)
if [ "$bold_files" -eq 0 ]; then
    print_error "No preprocessed BOLD files found"
    ((errors++))
else
    print_success "Found ${bold_files} preprocessed BOLD files"
fi

# Confounds
confound_files=$(find "mario.fmriprep/${SUBJECT}/${SESSION}/func" -name "*desc-confounds_timeseries.tsv" 2>/dev/null | wc -l)
if [ "$confound_files" -eq 0 ]; then
    print_error "No confound files found"
    ((errors++))
else
    print_success "Found ${confound_files} confound files"
fi

# Annotations
if [ ! -d "mario.annotations/annotations/${SUBJECT}/${SESSION}" ]; then
    print_warning "Annotations not found (may need manual download)"
else
    print_success "Annotations verified"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

print_header "Step 6/6: Installation Summary"

cd ..  # Return to tutorial root

echo "Installation location: $(pwd)/${SOURCEDATA_DIR}"
echo ""
echo "Directory structure:"
echo "  ${SOURCEDATA_DIR}/"
echo "  ├── mario/                     # Raw BIDS data"
echo "  ├── mario.fmriprep/            # Preprocessed fMRI"
echo "  ├── mario.annotations/         # Behavioral events"
echo "  ├── mario.scenes/              # Scene segmentation"
echo "  ├── mario.stimuli/             # Game ROM and files"
echo "  └── cneuromod.processed/       # Anatomical data"
echo ""

if [ $errors -eq 0 ]; then
    print_success "All downloads verified successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Run setup script: bash setup_environment.sh"
    echo "  2. Activate environment: source venv/bin/activate"
    echo "  3. Start Jupyter: jupyter notebook"
    echo "  4. Open notebooks/01_dataset_exploration.ipynb"
    echo ""
    print_info "Note: mario.replays derivatives need manual generation"
    print_info "      See mario.replays/README.md for instructions"
else
    print_warning "Installation completed with ${errors} error(s)"
    print_info "You may need to manually download some files"
fi

print_header "Installation Complete!"

echo -e "${GREEN}GL&HF! 🧠🎮${NC}"
