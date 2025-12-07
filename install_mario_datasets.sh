#!/bin/bash
# ==============================================================================
# Mario fMRI Tutorial - Data Download Script
# ==============================================================================
# This script downloads the CNeuromod Mario dataset for the tutorial
# Focus: Single subject (sub-01), single session (ses-010)
# Total download: ~7-8 GB
# Duration: ~10-15 minutes (depending on connection)
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

print_header "Step 1/7: Installing Dependencies"

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

print_header "Step 2/7: Creating Directory Structure"

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

print_header "Step 3/7: Installing Dataset Repositories"

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
    print_info "Installing mario with subdatasets..."
    datalad install --recursive git@github.com:courtois-neuromod/mario
    print_success "mario installed"
else
    print_warning "mario already exists"
fi

# stimuli (game ROM and integration files)
if [ ! -d "mario/stimuli" ]; then
    # Check if AWS credentials are available
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        print_warning "stimuli requires AWS credentials - skipping download"
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}STIMULI - AWS CREDENTIALS REQUIRED${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "The stimuli dataset contains the Super Mario Bros ROM and"
        echo "integration files. This data is stored on AWS S3 and requires"
        echo "access credentials."
        echo ""
        echo "Options to obtain access:"
        echo ""
        echo "  1. Request AWS credentials from the CNeuromod team:"
        echo "     • Contact: courtois-neuromod-admin@criugm.qc.ca"
        echo "     • Specify you need access to stimuli dataset"
        echo ""
        echo "  2. Obtain the ROM legally:"
        echo "     • Purchase Super Mario Bros for NES"
        echo "     • Extract ROM using legal methods"
        echo "     • Place in sourcedata/mario/stimuli/"
        echo ""
        echo "Note: The tutorial can proceed without stimuli, but some"
        echo "      visualizations and replay analyses will be limited."
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
    else
        print_info "Downloading stimuli files..."
        datalad get mario/stimuli
        print_success "stimuli downloaded"
    fi
else
    print_warning "mario/stimuli already exists"
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

# mario.replays (replay processing)
if [ ! -d "mario.replays" ]; then
    print_info "Installing mario.replays..."
    git clone git@github.com:courtois-neuromod/mario.replays
    print_success "mario.replays cloned"
else
    print_warning "mario.replays already exists"
fi

# ------------------------------------------------------------------------------
# Download Session Data
# ------------------------------------------------------------------------------

print_header "Step 4/7: Downloading Session Data (${SUBJECT}/${SESSION})"

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

print_header "Step 5/7: Verifying Downloads"

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
# Generate Derivatives
# ------------------------------------------------------------------------------

print_header "Step 6/7: Generating Derivatives for ${SUBJECT}/${SESSION}"

print_info "Generating derivatives from mario.replays, mario.scenes, and mario.annotations..."
echo ""

# Generate replays with game variables
if [ -d "${SOURCEDATA_DIR}/mario.replays" ]; then
    print_info "Setting up mario.replays environment and generating derivatives..."
    cd "${SOURCEDATA_DIR}/mario.replays"

    # Create environment and setup using invoke
    if [ ! -d "env" ]; then
        print_info "Creating virtual environment and installing dependencies..."
        python3 -m venv env && source env/bin/activate && invoke setup-env
        print_success "Environment setup complete"
    else
        print_info "Environment already exists, activating..."
        source env/bin/activate
    fi

    # Generate replay derivatives
    print_info "Generating replay derivatives for ${SUBJECT} ${SESSION}..."
    invoke create-replays \
        --subjects "${SUBJECT}" \
        --sessions "${SESSION}" \
        --datapath "../mario" \
        --save-confs \
        --save-variables || print_warning "Failed to generate replays"
    print_success "Replay derivatives generated in mario.replays/${SUBJECT}/"

    deactivate
    cd ../..
else
    print_warning "mario.replays not found - skipping replay generation"
fi

# Generate scene clips
if [ -d "${SOURCEDATA_DIR}/mario.scenes" ] && [ -d "${SOURCEDATA_DIR}/mario.replays/${SUBJECT}" ]; then
    print_info "Setting up mario.scenes environment and generating clips..."
    cd "${SOURCEDATA_DIR}/mario.scenes"

    # Create environment and setup using invoke
    if [ ! -d "env" ]; then
        print_info "Creating virtual environment and installing dependencies..."
        python3 -m venv env && source env/bin/activate && invoke setup-env && invoke get-scenes-data
        print_success "Environment setup complete"
    else
        print_info "Environment already exists, activating..."
        source env/bin/activate
    fi

    # Generate scene clips
    print_info "Generating scene clips for ${SUBJECT} ${SESSION}..."
    invoke create-clips \
        --subjects "${SUBJECT}" \
        --sessions "${SESSION}" \
        --save-videos \
        --save-variables || print_warning "Failed to generate scene clips"
    print_success "Scene clips generated in mario.scenes/clips/"

    deactivate
    cd ../..
else
    print_warning "mario.scenes not found or replays not generated - skipping scene clip generation"
fi

# Generate annotated events
if [ -d "${SOURCEDATA_DIR}/mario.annotations" ] && [ -d "${SOURCEDATA_DIR}/mario.replays/${SUBJECT}" ]; then
    print_info "Setting up mario.annotations environment and generating annotations..."
    cd "${SOURCEDATA_DIR}/mario.annotations"

    # Create environment and setup using invoke
    if [ ! -d "env" ]; then
        print_info "Creating virtual environment and installing dependencies..."
        python3 -m venv env && source env/bin/activate && invoke setup-env
        print_success "Environment setup complete"
    else
        print_info "Environment already exists, activating..."
        source env/bin/activate
    fi

    # Generate annotated events
    print_info "Generating annotated events for ${SUBJECT} ${SESSION}..."
    invoke generate-annotations \
        --datapath "../mario" \
        --subjects "${SUBJECT}" \
        --sessions "${SESSION}" || print_warning "Failed to generate annotations"
    print_success "Annotated events generated in mario.annotations/annotations/"

    deactivate
    cd ../..
else
    print_warning "mario.annotations not found or replays not generated - skipping annotation generation"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

print_header "Step 7/7: Installation Summary"

echo "Installation location: $(pwd)/${SOURCEDATA_DIR}"
echo ""
echo "Directory structure:"
echo "  ${SOURCEDATA_DIR}/"
echo "  ├── mario/                          # Raw BIDS data"
echo "  │   └── stimuli/                    # Game ROM and files (if AWS credentials provided)"
echo "  ├── mario.fmriprep/                 # Preprocessed fMRI"
echo "  ├── mario.annotations/"
echo "  │   └── annotations/                # Annotated event files (generated)"
echo "  │       └── ${SUBJECT}/${SESSION}/func/"
echo "  ├── mario.scenes/"
echo "  │   └── clips/                      # Scene clip videos (generated)"
echo "  ├── mario.replays/"
echo "  │   └── ${SUBJECT}/${SESSION}/beh/  # Replay derivatives with game variables (generated)"
echo "  └── cneuromod.processed/            # Anatomical data"
echo ""

if [ $errors -eq 0 ]; then
    print_success "All downloads and derivatives generated successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate main environment: source env/bin/activate"
    echo "  2. Start Jupyter: jupyter notebook"
    echo "  3. Open notebooks/01_dataset_exploration.ipynb"
    echo ""
    echo "Note: Each Mario package (replays, scenes, annotations) has its own 'env' folder"
    echo "      for isolated dependency management."
    echo ""

    # Check if derivatives were generated
    if [ -d "${SOURCEDATA_DIR}/mario.replays/${SUBJECT}" ] && [ -d "${SOURCEDATA_DIR}/mario.annotations/annotations" ]; then
        print_success "Derivatives ready in their respective repos"
    else
        print_warning "Some derivatives may not have been generated"
        print_info "You can generate them manually:"
        print_info "  cd ${SOURCEDATA_DIR}/mario.replays && source env/bin/activate && invoke create-replays --save-variables --subjects ${SUBJECT} --sessions ${SESSION}"
        print_info "  cd ${SOURCEDATA_DIR}/mario.scenes && source env/bin/activate && invoke create-clips --subjects ${SUBJECT} --sessions ${SESSION}"
        print_info "  cd ${SOURCEDATA_DIR}/mario.annotations && source env/bin/activate && invoke generate-annotations --subjects ${SUBJECT} --sessions ${SESSION}"
    fi
else
    print_warning "Installation completed with ${errors} error(s)"
    print_info "You may need to manually download some files or generate derivatives"
fi

print_header "Installation Complete!"

echo -e "${GREEN}GL&HF! 🧠🎮${NC}"
