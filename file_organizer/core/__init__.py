"""
Core module for file organization functionality.

This module contains the main FileOrganizer class and related functionality
for organizing files by various criteria.
"""

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

