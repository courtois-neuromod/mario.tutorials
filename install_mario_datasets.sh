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
SESSION="ses-010"             # primary session used by notebooks 00/01
SESSIONS=("ses-010" "ses-001") # all sessions to pre-fetch (notebook 03 uses ses-001 too)
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

# mario (raw BIDS data + annotations + gamelogs, on branch dev_replays)
if [ ! -d "mario" ]; then
    print_info "Installing mario with subdatasets (branch: dev_replays)..."
    datalad install --recursive git@github.com:courtois-neuromod/mario
    (cd mario && git checkout dev_replays)
    print_success "mario installed @ dev_replays"
else
    print_warning "mario already exists"
    (cd mario && git checkout dev_replays 2>/dev/null) || print_warning "Could not switch to dev_replays"
fi

# stimuli (game ROM + save states) — initialized by `--recursive` above, content fetched here.
# The public conp-ria-storage-http remote serves these; no AWS credentials needed.
if [ -f "mario/stimuli/SuperMarioBros-Nes/rom.nes" ]; then
    print_success "mario/stimuli ROM already present"
else
    print_info "Fetching mario/stimuli ROM + save states via datalad get..."
    if (cd mario && datalad get stimuli/); then
        print_success "stimuli content downloaded"
    else
        print_warning "Could not fetch stimuli content"
        echo "The Super Mario Bros ROM could not be downloaded automatically."
        echo "You can either:"
        echo "  • Retry later (transient network issue)"
        echo "  • Obtain SuperMarioBros.nes legally and drop it into sourcedata/mario/stimuli/SuperMarioBros-Nes/rom.nes"
        echo "Notebooks 00/01 work without it; notebook 02 (RL agent) needs the ROM."
    fi
fi

# mario.fmriprep (preprocessed fMRI)
if [ ! -d "mario.fmriprep" ]; then
    print_info "Installing mario.fmriprep..."
    datalad install git@github.com:courtois-neuromod/mario.fmriprep
    print_success "mario.fmriprep installed"
else
    print_warning "mario.fmriprep already exists"
fi

# Note: mario.annotations and mario.replays are no longer separate repos.
# Their contents now live under mario/sub-*/ses-*/func/*_desc-annotated_events.tsv
# and mario/sub-*/ses-*/gamelogs/* on the dev_replays branch.

# mario.scenes (scene segmentation — still an independent repo)
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

print_header "Step 4/6: Downloading Session Data for ${SUBJECT}"

print_info "Fetching sessions: ${SESSIONS[*]}"
echo ""

# Raw BIDS + annotations + gamelogs + preprocessed BOLD + confounds, per session
for SES in "${SESSIONS[@]}"; do
    print_info "── ${SES} ──"

    print_info "  Events + annotated events..."
    datalad get "mario/${SUBJECT}/${SES}/func"/*.tsv 2>/dev/null \
        || print_warning "  Some TSV files may not exist for ${SES}"

    print_info "  Gamelogs (replays, summaries, variables, lowlevel features, recordings)..."
    datalad get "mario/${SUBJECT}/${SES}/gamelogs"/* 2>/dev/null \
        || print_warning "  Some gamelogs files may not exist for ${SES}"

    print_info "  Preprocessed BOLD + confounds (this may take a few minutes)..."
    datalad get "mario.fmriprep/${SUBJECT}/${SES}/func"/*space-MNI152NLin2009cAsym* 2>/dev/null \
        || print_warning "  Some fMRIPrep files may not exist for ${SES}"
    datalad get "mario.fmriprep/${SUBJECT}/${SES}/func"/*desc-confounds* 2>/dev/null \
        || print_warning "  Some confounds may not exist for ${SES}"
done
print_success "Mario + fMRIPrep data downloaded for all sessions"

# Anatomical data (subject-level, session-independent)
print_info "Downloading anatomical data..."
datalad get "cneuromod.processed/smriprep/${SUBJECT}/anat"/*space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz 2>/dev/null || print_warning "Anatomical may not exist"
datalad get "cneuromod.processed/smriprep/${SUBJECT}/anat"/*space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz 2>/dev/null || print_warning "Brain mask may not exist"
print_success "Anatomical data downloaded"

# ------------------------------------------------------------------------------
# Verify Downloads
# ------------------------------------------------------------------------------

print_header "Step 5/6: Verifying Downloads"

# Check for key files across all fetched sessions
errors=0

for SES in "${SESSIONS[@]}"; do
    print_info "── verifying ${SES} ──"

    if [ ! -f "mario/${SUBJECT}/${SES}/func/${SUBJECT}_${SES}_task-mario_run-01_events.tsv" ]; then
        print_error "  Raw events file not found for ${SES}"
        ((errors++))
    else
        print_success "  Raw events verified"
    fi

    bold_files=$(find "mario.fmriprep/${SUBJECT}/${SES}/func" -name "*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" 2>/dev/null | wc -l)
    if [ "$bold_files" -eq 0 ]; then
        print_error "  No preprocessed BOLD files found for ${SES}"
        ((errors++))
    else
        print_success "  Found ${bold_files} preprocessed BOLD files"
    fi

    confound_files=$(find "mario.fmriprep/${SUBJECT}/${SES}/func" -name "*desc-confounds_timeseries.tsv" 2>/dev/null | wc -l)
    if [ "$confound_files" -eq 0 ]; then
        print_error "  No confound files found for ${SES}"
        ((errors++))
    else
        print_success "  Found ${confound_files} confound files"
    fi

    if [ ! -f "mario/${SUBJECT}/${SES}/func/${SUBJECT}_${SES}_task-mario_run-01_desc-annotated_events.tsv" ]; then
        print_error "  Annotated events TSV not found for ${SES}"
        ((errors++))
    else
        print_success "  Annotated events verified"
    fi
done

# Lowlevel features (now in mario/sub-XX/ses-YYY/gamelogs/)
lowlevel_count=$(find "mario/${SUBJECT}/${SESSION}/gamelogs" -name "*_lowlevel.npy" 2>/dev/null | wc -l)
if [ "$lowlevel_count" -eq 0 ]; then
    print_error "No lowlevel feature files found in gamelogs/"
    ((errors++))
else
    print_success "Found ${lowlevel_count} lowlevel feature files"
fi

# Summary JSONs (now in mario/sub-XX/ses-YYY/gamelogs/)
summary_count=$(find "mario/${SUBJECT}/${SESSION}/gamelogs" -name "*_summary.json" 2>/dev/null | wc -l)
if [ "$summary_count" -eq 0 ]; then
    print_warning "No replay summary JSONs found in gamelogs/"
else
    print_success "Found ${summary_count} replay summary files"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

print_header "Step 6/6: Installation Summary"

echo "Installation location: $(pwd)/${SOURCEDATA_DIR}"
echo ""
echo "Directory structure:"
echo "  ${SOURCEDATA_DIR}/"
echo "  ├── mario/                              # Raw BIDS + annotations + gamelogs (branch: dev_replays)"
echo "  │   ├── ${SUBJECT}/${SESSION}/"
echo "  │   │   ├── func/"
echo "  │   │   │   ├── *_events.tsv                # Raw event timing"
echo "  │   │   │   └── *_desc-annotated_events.tsv # Rich annotated events"
echo "  │   │   └── gamelogs/"
echo "  │   │       ├── *.bk2                        # Replay files"
echo "  │   │       ├── *_summary.json               # Replay metadata"
echo "  │   │       ├── *_variables.json             # Game variables"
echo "  │   │       ├── *_lowlevel.npy               # Luminance / optical flow / audio"
echo "  │   │       └── *_recording.mp4              # Video"
echo "  │   └── stimuli/                        # Game ROM (if AWS credentials provided)"
echo "  ├── mario.fmriprep/                     # Preprocessed fMRI"
echo "  ├── mario.scenes/                       # Scene segmentation (cloned, not generated)"
echo "  └── cneuromod.processed/                # Anatomical data"
echo ""
print_info "Note: Annotations and replay derivatives are pre-shipped on the dev_replays branch."
print_info "To regenerate from scratch, see mario/code/annotations/ and mario/code/replays/."
echo ""

if [ $errors -eq 0 ]; then
    print_success "All downloads verified successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Activate main environment: source env/bin/activate"
    echo "  2. Start Jupyter: jupyter notebook"
    echo "  3. Open notebooks/00_dataset_overview.ipynb"
    echo ""
else
    print_warning "Installation completed with ${errors} error(s)"
    print_info "You may need to manually download some files"
fi

print_header "Installation Complete!"

echo -e "${GREEN}GL&HF! 🧠🎮${NC}"
