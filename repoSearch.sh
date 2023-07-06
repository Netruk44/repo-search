#! /bin/bash

VENV_DIR="$(dirname "$0")/.venv"
REQUIREMENTS_FILE="$(dirname "$0")/requirements.txt"

# Check if 'init' or 'clean' was passed as an argument
if [ "$1" == "init" ] || [ ! -d "$VENV_DIR" ]; then
    
    # Check if the venv already exists
    if [ -d "$VENV_DIR" ]; then
        echo "ERROR: Virtual environment already exists." >&2
        exit 1
    fi

    echo "Initializing virtual environment..." >&2
    python3 -m venv "$VENV_DIR" >&2
    source "$VENV_DIR/bin/activate" >&2
    echo "Installing requirements..." >&2
    pip install -r "$REQUIREMENTS_FILE" >&2
    deactivate >&2
    echo "Init completed, ready to run." >&2

    # Exit if 'init', otherwise continue to execute the script
    if [ "$1" == "init" ]; then
        exit 0
    fi
elif [ "$1" == "clean" ]; then
    echo "Cleaning virtual environment..."
    echo "Removing $VENV_DIR..."
    rm -rf "$VENV_DIR"
    exit 0
fi

# Activate the venv and start the interface script
source "$VENV_DIR/bin/activate"
python3 "$(dirname "$0")/interface.py" "$@"
deactivate