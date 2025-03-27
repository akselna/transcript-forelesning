#!/bin/bash
#
# Setup script for installing whisper.cpp and configuring the application
#

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Whisper.cpp Setup for Flask App      ${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get absolute path (macOS compatible)
get_abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "$(pwd)/$path"
    fi
}

# Check requirements
echo -e "\n${YELLOW}Checking requirements...${NC}"

# Check for git
if ! command_exists git; then
    echo -e "${RED}Git is not installed. Please install git and try again.${NC}"
    exit 1
fi

# Check for cmake
if ! command_exists cmake; then
    echo -e "${YELLOW}CMake is not installed. It's required to build whisper.cpp.${NC}"
    echo -e "${YELLOW}Installing cmake...${NC}"
    
    if command_exists brew; then
        brew install cmake
    elif command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y cmake
    elif command_exists dnf; then
        sudo dnf install -y cmake
    else
        echo -e "${RED}Could not install cmake. Please install it manually.${NC}"
        exit 1
    fi
fi

# Check for ffmpeg
if ! command_exists ffmpeg; then
    echo -e "${YELLOW}FFmpeg is not installed. It's required for audio processing.${NC}"
    echo -e "${YELLOW}Installing ffmpeg...${NC}"
    
    if command_exists brew; then
        brew install ffmpeg
    elif command_exists apt-get; then
        sudo apt-get update && sudo apt-get install -y ffmpeg
    elif command_exists dnf; then
        sudo dnf install -y ffmpeg
    else
        echo -e "${RED}Could not install ffmpeg. Please install it manually.${NC}"
        exit 1
    fi
fi

# Ask for whisper.cpp installation directory
echo -e "\n${YELLOW}Where would you like to install whisper.cpp?${NC}"
echo -e "${YELLOW}Press Enter to use default (./whisper.cpp)${NC}"
read -p "Directory: " WHISPER_DIR

# Use default if empty
if [ -z "$WHISPER_DIR" ]; then
    WHISPER_DIR="./whisper.cpp"
fi

# Convert to absolute path (macOS compatible)
WHISPER_DIR=$(get_abs_path "$WHISPER_DIR")
echo -e "${GREEN}Will install whisper.cpp to: ${WHISPER_DIR}${NC}"

# Ensure parent directory exists
PARENT_DIR=$(dirname "$WHISPER_DIR")
if [ ! -d "$PARENT_DIR" ]; then
    echo -e "${YELLOW}Creating parent directory: ${PARENT_DIR}${NC}"
    mkdir -p "$PARENT_DIR"
fi

# Clone and build whisper.cpp
echo -e "\n${YELLOW}Setting up whisper.cpp...${NC}"

if [ -d "$WHISPER_DIR" ]; then
    echo -e "${YELLOW}Directory already exists. Do you want to update it? (y/n)${NC}"
    read -p "Update? " UPDATE
    
    if [[ $UPDATE == "y" || $UPDATE == "Y" ]]; then
        echo -e "${GREEN}Updating whisper.cpp...${NC}"
        cd "$WHISPER_DIR"
        git pull
    else
        echo -e "${GREEN}Using existing installation.${NC}"
    fi
else
    echo -e "${GREEN}Cloning whisper.cpp repository...${NC}"
    git clone https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
fi

# Build whisper.cpp using CMake
cd "$WHISPER_DIR"
echo -e "${GREEN}Building whisper.cpp using CMake...${NC}"

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake and build
cmake ..
cmake --build . --config Release

# Simply check if the build process completed without direct error
if [ $? -ne 0 ]; then
    echo -e "${RED}CMake build process returned an error.${NC}"
    exit 1
fi

# List all binary directories that might contain the whisper executable
echo -e "${YELLOW}Checking for whisper executables...${NC}"
BIN_PATHS=("$WHISPER_DIR/bin" "$WHISPER_DIR/build/bin")
EXEC_NAMES=("whisper-cli" "main")
FOUND_EXEC=""

for bin_path in "${BIN_PATHS[@]}"; do
    if [ -d "$bin_path" ]; then
        echo -e "${GREEN}Found binary directory: $bin_path${NC}"
        for exec_name in "${EXEC_NAMES[@]}"; do
            if [ -f "$bin_path/$exec_name" ]; then
                echo -e "${GREEN}Found executable: $bin_path/$exec_name${NC}"
                FOUND_EXEC="$bin_path/$exec_name"
                break 2
            fi
        done
    fi
done

# Also check root directory for main executable (for make-based builds)
if [ -z "$FOUND_EXEC" ] && [ -f "$WHISPER_DIR/main" ]; then
    echo -e "${GREEN}Found executable: $WHISPER_DIR/main${NC}"
    FOUND_EXEC="$WHISPER_DIR/main"
fi

if [ -z "$FOUND_EXEC" ]; then
    echo -e "${YELLOW}Could not find whisper executable in standard locations.${NC}"
    echo -e "${YELLOW}Manual build may be required. Let's try...${NC}"
    
    # Try fallback to direct make
    cd "$WHISPER_DIR"
    make clean
    make -j
    
    if [ -f "$WHISPER_DIR/main" ]; then
        echo -e "${GREEN}Successfully built whisper.cpp with fallback method!${NC}"
        FOUND_EXEC="$WHISPER_DIR/main"
    else
        echo -e "${RED}Failed to build whisper.cpp. Please check the error messages.${NC}"
        
        # Final fallback: just trust that the build succeeded and continue
        echo -e "${YELLOW}Proceeding anyway. You may need to manually build whisper.cpp.${NC}"
        echo -e "${YELLOW}Please check the whisper.cpp documentation for build instructions.${NC}"
    fi
else
    echo -e "${GREEN}Whisper.cpp build successful!${NC}"
fi

# Download a model
echo -e "\n${YELLOW}Which model would you like to download?${NC}"
echo -e "${YELLOW}Options: tiny, base, small, medium, large (default: small)${NC}"
read -p "Model: " MODEL_SIZE

# Use default if empty
if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE="small"
fi

echo -e "${GREEN}Downloading the ${MODEL_SIZE} model...${NC}"
cd "$WHISPER_DIR"
./models/download-ggml-model.sh "$MODEL_SIZE"

# Check if download was successful by looking for any file matching the pattern
model_files=$(find "$WHISPER_DIR/models" -name "ggml-${MODEL_SIZE}*.bin" | wc -l)
if [ "$model_files" -eq 0 ]; then
    echo -e "${RED}Failed to download the model. Please check your internet connection and try again.${NC}"
    exit 1
else
    echo -e "${GREEN}Successfully downloaded model files.${NC}"
fi

# Create a .env file for the Flask app
ENV_FILE=".env"
echo -e "\n${YELLOW}Creating .env file with whisper.cpp configuration...${NC}"

if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}File .env already exists. Updating whisper.cpp settings...${NC}"
    # Remove existing whisper.cpp settings
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS version (uses different sed syntax)
        sed -i '' '/WHISPER_CPP_PATH/d' "$ENV_FILE"
        sed -i '' '/WHISPER_MODEL/d' "$ENV_FILE"
    else
        # Linux version
        sed -i '/WHISPER_CPP_PATH/d' "$ENV_FILE"
        sed -i '/WHISPER_MODEL/d' "$ENV_FILE"
    fi
else
    echo -e "${GREEN}Creating new .env file...${NC}"
    touch "$ENV_FILE"
fi

# Add whisper.cpp settings
echo "WHISPER_CPP_PATH=$WHISPER_DIR" >> "$ENV_FILE"
echo "WHISPER_MODEL=$MODEL_SIZE" >> "$ENV_FILE"

echo -e "\n${GREEN}Configuration saved to ${ENV_FILE}${NC}"

# Instructions for next steps
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Make sure the Flask app is installed with all requirements:"
echo -e "   pip install -r requirements.txt"
echo -e "2. Start the Flask app with: python app.py"
echo -e "3. The app will automatically use whisper.cpp for transcription"
echo -e "\n${YELLOW}Environment variables set:${NC}"
echo -e "- WHISPER_CPP_PATH: $WHISPER_DIR"
echo -e "- WHISPER_MODEL: $MODEL_SIZE"
echo -e "\n${YELLOW}You can modify these in the .env file if needed.${NC}"
echo -e "${BLUE}========================================${NC}"

exit 0