# NLP
NLP - Group 19 Repository



# Project Setup Guide

This guide explains how to create and work with a Python virtual environment and how to open Visual Studio Code (VS Code) for your project.

## Virtual Environment Setup

### Prerequisites

- **Python:** Ensure Python is installed. Check by running:
  ```bash
  python --version
  ```
- **pip:** Make sure pip is updated to install packages smoothly.

### Activating the Virtual Environment

- **Linux (using WSL through cmd):**
  Go to repository directory and run:
  ```bash
  source env/bin/activate
  ```

After activation, your terminal prompt should display the environment name (`VENV_NLP`).

### Deactivating the Virtual Environment

When you're finished working, deactivate the environment by running:
```bash
deactivate
```

## Opening VS Code

### Launching VS Code from the Terminal

1. Navigate to your project directory:
   ```bash
   cd path/to/your/project
   ```

2. Activate the virtual environment

3. Open VS Code by running:
   ```bash
   code .
   ```
   This command opens the current directory in VS Code.

### Additional VS Code Tips

- **Integrated Terminal:**  
  Use VS Code's integrated terminal via **View > Terminal** to work within your virtual environment seamlessly.
- **Extensions:**  
  Install the Python extension for enhanced development features such as IntelliSense, linting, and debugging.

