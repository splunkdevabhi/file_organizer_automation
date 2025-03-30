# File Organizer Automation

A Python-based utility that automates file organization tasks to help keep your directories clean and structured. This tool can monitor directories, sort files based on extensions, apply naming conventions, and automate organization at system startup.

## Features

- **Automatic File Sorting**: Organizes files based on file extensions or custom rules
- **Directory Monitoring**: Watches specified directories for changes and organizes files in real-time
- **Custom Organization Rules**: Define your own organization logic through configuration files
- **Platform Support**: Works on macOS (using launchd) and Linux (using systemd)
- **Startup Automation**: Can be configured to run automatically at system login
- **Logging**: Comprehensive logging of all file operations
- **Command-line Interface**: Easy-to-use commands for installation and configuration

## Installation

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Install from GitHub
```bash
# Clone the repository
git clone https://github.com/splunkdevabhi/file_organizer_automation.git

# Navigate to the project directory
cd file_organizer_automation

# Install the package
pip install -e .
```

### Install from PyPI
```bash
pip install file-organizer-automation
```

## Usage Examples

### Basic Usage
```python
from file_organizer.core.organizer import FileOrganizer

# Initialize the organizer
organizer = FileOrganizer('/path/to/directory')

# Organize files by extension
organizer.organize_by_extension()

# Organize files by creation date
organizer.organize_by_date()
```

### Command Line Interface
```bash
# Organize a specific directory
file-organizer --path /path/to/directory

# Setup automated organization at login
file-organizer --setup-startup

# Remove startup automation
file-organizer --remove-startup

# Run with a custom configuration file
file-organizer --config /path/to/config.json
```

## Project Structure
```
file_organizer_automation/
├── file_organizer/               # Main package
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   └── organizer.py          # Main organizer class
│   ├── utils/                    # Utility functions
│   │   └── __init__.py
│   ├── scripts/                  # Scripts for automation
│   │   └── startup.py            # Startup automation script
│   └── __init__.py
├── tests/                        # Test suite
├── docs/                         # Documentation
├── setup.py                      # Package installation configuration
├── .gitignore                    # Git ignore file
├── LICENSE                       # License file
└── README.md                     # This file
```

## Dependencies

- **Python 3.6+**: Core language
- **watchdog**: For real-time file system monitoring
- **schedule**: For scheduling organization tasks
- **click**: For command-line interface
- **appdirs**: For determining platform-specific directories

## Configuration

The file organizer can be configured using a JSON configuration file:

```json
{
  "directories": {
    "watch": [
      "~/Downloads",
      "~/Desktop"
    ],
    "exclude": [
      "~/Downloads/Important"
    ]
  },
  "rules": [
    {
      "extensions": ["jpg", "png", "gif", "bmp"],
      "destination": "~/Pictures"
    },
    {
      "extensions": ["doc", "docx", "pdf", "txt"],
      "destination": "~/Documents"
    },
    {
      "extensions": ["mp3", "wav", "flac"],
      "destination": "~/Music"
    }
  ],
  "options": {
    "create_subdirectories": true,
    "preserve_original_name": true,
    "date_format": "%Y-%m-%d"
  }
}
```

## Contributing Guidelines

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

### Coding Standards
- Follow PEP 8 style guide
- Include docstrings for all functions, classes, and modules
- Write unit tests for new features
- Keep functions small and focused on a single responsibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

