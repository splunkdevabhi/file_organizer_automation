"""
File Organizer - A tool for organizing files by various criteria.

This package provides functionality to organize files by extension, date,
and other criteria, as well as monitoring directories for changes.
"""

__version__ = "0.1.0"
__author__ = "Abhishek Shukla"

# Expose main classes and functions at package level
from file_organizer.core.organizer import (
    FileOrganizer,
    organize_by_extension,
    organize_by_date,
    organize_duplicates,
)

# Define what's available on import
__all__ = [
    "FileOrganizer",
    "organize_by_extension",
    "organize_by_date",
    "organize_duplicates",
]

