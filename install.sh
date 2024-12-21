#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda before running this script."
    exit 1
fi

# Check if the conda environment exists
if conda env list | grep -q "^diffvis "; then
    echo "Activating existing Conda environment: diffvis"
    source activate diffvis
else
    echo "Creating Conda environment: diffvis"
    conda create -y -n diffvis python=3.11
    source activate diffvis
fi

# Export PATH for the active Conda environment
export PATH="$(conda info --base)/envs/diffvis/bin:$PATH"

# Install dependencies
pip install -r requirements.txt
python install -e .

# Install DFlat package
cd DFlat
pip install .
cd ..
