#!/bin/bash
# Setup script for LLM integration (optional)

echo "=== Flood Risk Viewer - LLM Setup ==="
echo ""
echo "This tool can use AI to generate dynamic flood risk explanations."
echo "If you don't configure LLM access, it will use rule-based explanations (still good!)."
echo ""

read -p "Do you want to enable LLM-generated explanations? (y/n): " enable_llm

if [[ "$enable_llm" != "y" ]]; then
    echo "✓ Skipping LLM setup. Rule-based explanations will be used."
    exit 0
fi

echo ""
echo "Choose your LLM provider:"
echo "  1) OpenAI (GPT-4, recommended)"
echo "  2) Anthropic (Claude)"
echo ""
read -p "Enter choice (1 or 2): " provider

if [[ "$provider" == "1" ]]; then
    echo ""
    echo "OpenAI API Key setup:"
    echo "  1. Get your API key from: https://platform.openai.com/api-keys"
    echo "  2. Paste it below (it will be added to your .bashrc or .zshrc)"
    echo ""
    read -p "Enter OpenAI API key: " api_key 
    
    if [[ -n "$api_key" ]]; then
        # Detect shell
        if [[ -n "$ZSH_VERSION" ]]; then
            shell_rc="$HOME/.zshrc"
        else
            shell_rc="$HOME/.bashrc"
        fi
        
        echo "" >> "$shell_rc"
        echo "# Flood Risk Viewer - OpenAI API" >> "$shell_rc"
        echo "export OPENAI_API_KEY=\"$api_key\"" >> "$shell_rc"
        
        export OPENAI_API_KEY="$api_key"
        
        echo "✓ Added to $shell_rc"
        echo "✓ Run 'source $shell_rc' or restart your terminal"
        
        # Install OpenAI package if not present
        pip3 list | grep -q openai || {
            echo ""
            read -p "Install openai package? (y/n): " install_pkg
            if [[ "$install_pkg" == "y" ]]; then
                pip3 install openai
                echo "✓ Installed openai package"
            fi
        }
    fi

elif [[ "$provider" == "2" ]]; then
    echo ""
    echo "Anthropic API Key setup:"
    echo "  1. Get your API key from: https://console.anthropic.com/settings/keys"
    echo "  2. Paste it below"
    echo ""
    read -p "Enter Anthropic API key: " api_key
    
    if [[ -n "$api_key" ]]; then
        if [[ -n "$ZSH_VERSION" ]]; then
            shell_rc="$HOME/.zshrc"
        else
            shell_rc="$HOME/.bashrc"
        fi
        
        echo "" >> "$shell_rc"
        echo "# Flood Risk Viewer - Anthropic API" >> "$shell_rc"
        echo "export ANTHROPIC_API_KEY=\"$api_key\"" >> "$shell_rc"
        
        export ANTHROPIC_API_KEY="$api_key"
        
        echo "✓ Added to $shell_rc"
        echo "✓ Run 'source $shell_rc' or restart your terminal"
        
        # Install Anthropic package if not present
        pip3 list | grep -q anthropic || {
            echo ""
            read -p "Install anthropic package? (y/n): " install_pkg
            if [[ "$install_pkg" == "y" ]]; then
                pip3 install anthropic
                echo "✓ Installed anthropic package"
            fi
        }
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo "Run 'streamlit run ui.py' to test LLM explanations."
