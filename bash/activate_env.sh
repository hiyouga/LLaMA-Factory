#!/bin/bash

# LLaMA-Factory Environment Activation Helper
# This script helps activate and use the installed environment
# Created: 2025-09-21

echo "🔧 LLaMA-Factory Environment Helper"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run ./bash/install_env.sh first"
    exit 1
fi

echo "✅ Virtual environment found at .venv/"
echo ""
echo "To activate the environment, run:"
echo "   source .venv/bin/activate"
echo ""
echo "Or run this script with 'source' to activate automatically:"
echo "   source bash/activate_env.sh"
echo ""

# If script is sourced, activate the environment
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "🚀 Activating virtual environment..."
    source .venv/bin/activate
    echo "✅ Environment activated!"
    echo ""
    echo "You can now use:"
    echo "   • python (with llamafactory available)"
    echo "   • llamafactory-cli [command]"
    echo ""
    echo "To deactivate later, run: deactivate"
fi
