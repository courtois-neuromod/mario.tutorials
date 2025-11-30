echo "Installing mario datasets in $(pwd)"

mkdir data
cd data
datalad install git@github.com:courtois-neuromod/cneuromod.processed
datalad install git@github.com:courtois-neuromod/mario
datalad install git@github.com:courtois-neuromod/mario.fmriprep
datalad install git@github.com:courtois-neuromod/mario.annotations
datalad install git@github.com:courtois-neuromod/mario.scenes
datalad install git@github.com:courtois-neuromod/mario.replays

subject="sub-01"
datalad get "mario/${subject}/ses-010/func"/*.tsv
datalad get "mario/${subject}/ses-010/gamelogs"/*.bk2
datalad get "mario.fmriprep/${subject}/ses-010/func"/*space-MNI152NLin2009cAsym*
datalad get "mario.fmriprep/${subject}/ses-010/func"/*desc-confounds*
cd ..
