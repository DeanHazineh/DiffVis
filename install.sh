#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda before running this script."
    exit 1
fi

# Check if the conda environment exists
if conda env list | grep -q "^diffvis "; then
    echo "Activating existing Conda environment: diffvis"
    conda activate diffvis
else
    echo "Creating Conda environment: diffvis"
    conda create -y -n diffvis python=3.11
    conda activate diffvis
fi

# Export PATH for the active Conda environment
export PATH="$(conda info --base)/envs/diffvis/bin:$PATH"

# Install PyTorch with CUDA support
echo "Installing PyTorch with GPU support..."
# Replace 'cu118' with the appropriate CUDA version if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing additional dependencies..."
pip install -r requirements.txt
python install -e .

# Install DFlat package
echo "Installing DFlat package..."
cd DFlat
pip install .
cd ..

echo "Installation complete."
