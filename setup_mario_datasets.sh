echo "Installing mario datasets in $(pwd)"

pip install datalad invoke airoh

# Create sourcedata directory
mkdir sourcedata
cd sourcedata

# Download repositories
datalad install git@github.com:courtois-neuromod/cneuromod.processed
datalad install git@github.com:courtois-neuromod/mario
cd mario
datalad install git@github.com:courtois-neuromod/mario.stimuli
cd mario.stimuli
datalad get .
cd ..
datalad install git@github.com:courtois-neuromod/mario.fmriprep
git clone git@github.com:courtois-neuromod/mario.annotations
git clone git@github.com:courtois-neuromod/mario.scenes
git clone git@github.com:courtois-neuromod/mario.replays

# Download example data for one subject
subject="sub-01"
datalad get "mario/${subject}/ses-010/func"/*.tsv
datalad get "mario/${subject}/ses-010/gamelogs"/*.bk2
datalad get "mario.fmriprep/${subject}/ses-010/func"/*space-MNI152NLin2009cAsym*
datalad get "mario.fmriprep/${subject}/ses-010/func"/*desc-confounds*

# Use replay tool to generate behavioral derivatives
cd mario.replays
invoke setup-env
source env/bin/activate
python code/mario_replays/create_replays/create_replays.py -d ../mario --save_videos --save_variables --save_states --save_ramdumps

cd ..
