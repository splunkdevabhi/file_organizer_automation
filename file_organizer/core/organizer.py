#!/usr/bin/env python3
"""
File Organizer Core Module

This module provides the core functionality for organizing files based on
different criteria like extension, date, etc. It also includes directory
monitoring capabilities and file operation utilities.
"""

import os
import sys
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union, Set
from threading import Thread
from functools import wraps
import click
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/.file_organizer/logs/file_organizer.log"))
    ]
)

# Create logger
logger = logging.getLogger("file_organizer")


def ensure_directory(func):
    """Decorator to ensure target directory exists before file operations."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not os.path.exists(self.target_dir):
            try:
                os.makedirs(self.target_dir)
                logger.info(f"Created target directory: {self.target_dir}")
            except OSError as e:
                logger.error(f"Failed to create target directory: {e}")
                raise
        return func(self, *args, **kwargs)
    return wrapper


class FileOrganizer:
    """
    FileOrganizer class that provides methods for organizing files
    based on various criteria and monitoring directories for changes.
    """

    def __init__(self, source_dir: str, target_dir: str = None):
        """
        Initialize the FileOrganizer with source and target directories.

        Args:
            source_dir (str): Directory to organize or monitor
            target_dir (str, optional): Directory to move organized files to.
                                         If None, uses source_dir. Defaults to None.
        """
        self.source_dir = os.path.abspath(os.path.expanduser(source_dir))
        self.target_dir = os.path.abspath(os.path.expanduser(target_dir)) if target_dir else self.source_dir
        self.observer = None
        self.monitor_thread = None
        self.is_monitoring = False

        # Initialize log directory
        log_dir = os.path.expanduser("~/.file_organizer/logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        logger.info(f"Initialized FileOrganizer with source: {self.source_dir} and target: {self.target_dir}")

    @ensure_directory
    def organize_by_extension(self, extensions_to_ignore: List[str] = None) -> Dict[str, int]:
        """
        Organize files in the source directory by their extensions.

        Args:
            extensions_to_ignore (List[str], optional): List of extensions to ignore. Defaults to None.

        Returns:
            Dict[str, int]: Count of files organized by extension
        """
        extensions_to_ignore = extensions_to_ignore or []
        stats = {}

        try:
            # Convert to lowercase for case-insensitive comparison
            extensions_to_ignore = [ext.lower() for ext in extensions_to_ignore]
            
            for item in os.listdir(self.source_dir):
                item_path = os.path.join(self.source_dir, item)
                
                # Skip directories and non-files
                if not os.path.isfile(item_path):
                    continue
                
                # Get the file extension
                _, extension = os.path.splitext(item)
                extension = extension.lower()
                
                # Skip files with no extension or ignored extensions
                if not extension or extension[1:] in extensions_to_ignore:
                    continue
                
                # Create a directory for the extension if it doesn't exist
                extension_dir = os.path.join(self.target_dir, extension[1:])
                os.makedirs(extension_dir, exist_ok=True)
                
                # Move the file to the extension directory
                target_path = os.path.join(extension_dir, item)
                if os.path.exists(target_path):
                    # If the file already exists, append a timestamp
                    base, ext = os.path.splitext(item)
                    new_name = f"{base}_{int(time.time())}{ext}"
                    target_path = os.path.join(extension_dir, new_name)
                
                shutil.move(item_path, target_path)
                logger.info(f"Moved {item} to {extension_dir}")
                
                # Update stats
                ext_key = extension[1:]
                stats[ext_key] = stats.get(ext_key, 0) + 1
            
            logger.info(f"Organized {sum(stats.values())} files by extension")
            return stats
            
        except Exception as e:
            logger.error(f"Error organizing files by extension: {e}")
            raise

    @ensure_directory
    def organize_by_date(self, date_format: str = "%Y-%m-%d") -> Dict[str, int]:
        """
        Organize files in the source directory by their modification dates.

        Args:
            date_format (str, optional): Format for date directories. Defaults to "%Y-%m-%d".

        Returns:
            Dict[str, int]: Count of files organized by date
        """
        stats = {}
        
        try:
            for item in os.listdir(self.source_dir):
                item_path = os.path.join(self.source_dir, item)
                
                # Skip directories and non-files
                if not os.path.isfile(item_path):
                    continue
                
                # Get the file's modification time
                mod_time = os.path.getmtime(item_path)
                mod_date = datetime.fromtimestamp(mod_time)
                date_str = mod_date.strftime(date_format)
                
                # Create a directory for the date if it doesn't exist
                date_dir = os.path.join(self.target_dir, date_str)
                os.makedirs(date_dir, exist_ok=True)
                
                # Move the file to the date directory
                target_path = os.path.join(date_dir, item)
                if os.path.exists(target_path):
                    # If the file already exists, append a timestamp
                    base, ext = os.path.splitext(item)
                    new_name = f"{base}_{int(time.time())}{ext}"
                    target_path = os.path.join(date_dir, new_name)
                
                shutil.move(item_path, target_path)
                logger.info(f"Moved {item} to {date_dir}")
                
                # Update stats
                stats[date_str] = stats.get(date_str, 0) + 1
            
            logger.info(f"Organized {sum(stats.values())} files by date")
            return stats
            
        except Exception as e:
            logger.error(f"Error organizing files by date: {e}")
            raise

    @ensure_directory
    def organize_by_size(self, size_categories: Dict[str, int] = None) -> Dict[str, int]:
        """
        Organize files in the source directory by their sizes.

        Args:
            size_categories (Dict[str, int], optional): Size categories in bytes.
                Default categories: {'small': 1MB, 'medium': 10MB, 'large': 100MB, 'very_large': >100MB}

        Returns:
            Dict[str, int]: Count of files organized by size category
        """
        if size_categories is None:
            # Default size categories (in bytes)
            size_categories = {
                'small': 1024 * 1024,  # 1MB
                'medium': 10 * 1024 * 1024,  # 10MB
                'large': 100 * 1024 * 1024,  # 100MB
                'very_large': float('inf')  # Anything larger
            }
        
        stats = {'small': 0, 'medium': 0, 'large': 0, 'very_large': 0}
        
        try:
            # Create directories for each size category
            for category in size_categories:
                category_dir = os.path.join(self.target_dir, category)
                os.makedirs(category_dir, exist_ok=True)
            
            for item in os.listdir(self.source_dir):
                item_path = os.path.join(self.source_dir, item)
                
                # Skip directories and non-files
                if not os.path.isfile(item_path):
                    continue
                
                # Get the file size
                file_size = os.path.getsize(item_path)
                
                # Determine the size category
                category = 'very_large'  # Default to largest category
                for cat_name, cat_size in sorted(size_categories.items(), key=lambda x: x[1]):
                    if file_size <= cat_size:
                        category = cat_name
                        break
                
                # Move the file to the appropriate size directory
                category_dir = os.path.join(self.target_dir, category)
                target_path = os.path.join(category_dir, item)
                
                if os.path.exists(target_path):
                    # If the file already exists, append a timestamp
                    base, ext = os.path.splitext(item)
                    new_name = f"{base}_{int(time.time())}{ext}"
                    target_path = os.path.join(category_dir, new_name)
                
                shutil.move(item_path, target_path)
                logger.info(f"Moved {item} ({file_size} bytes) to {category_dir}")
                
                # Update stats
                stats[category] = stats.get(category, 0) + 1
            
            logger.info(f"Organized {sum(stats.values())} files by size")
            return stats
            
        except Exception as e:
            logger.error(f"Error organizing files by size: {e}")
            raise

    def start_monitoring(self, organize_method: str = 'extension', interval: int = 60) -> None:
        """
        Start monitoring the source directory for new files and organize them.

        Args:
            organize_method (str, optional): Method to use for organizing ('extension', 'date', 'size'). 
                                             Defaults to 'extension'.
            interval (int, optional): Monitoring interval in seconds. Defaults to 60.
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return

        # Map of organize methods to their corresponding class methods
        method_map = {
            'extension': self.organize_by_extension,
            'date': self.organize_by_date,
            'size': self.organize_by_size
        }

        if organize_method not in method_map:
            logger.error(f"Invalid organize method: {organize_method}")
            raise ValueError(f"Invalid organize method. Choose from: {', '.join(method_map.keys())}")

        organize_func = method_map[organize_method]
        
        class FileHandler(FileSystemEventHandler):
            def __init__(self, organizer_func):
                self.organizer_func = organizer_func
                self.last_processed = 0
                self.pending_events = False
                super().__init__()
            
            def on_any_event(self, event):
                if event.is_directory:
                    return
                
                if event.event_type in ['created', 'modified']:
                    self.pending_events = True
                    # Throttle processing to avoid excessive operations
                    current_time = time.time()
                    if current_time - self.last_processed > interval:
                        self.process_events()
            
            def process_events(self):
                if self.pending_events:
                    logger.info("Processing pending file events")
                    try:
                        self.organizer_func()
                        self.pending_events = False
                        self.last_processed = time.time()
                    except Exception as e:
                        logger.error(f"Error processing events: {e}")

        # Initialize and start the watchdog observer
        event_handler = FileHandler(organize_func)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.source_dir, recursive=False)
        self.observer.start()
        
        def monitor_loop():
            logger.info(f"Started monitoring {self.source_dir} using {organize_method} organization method")
            try:
                while self.is_monitoring:
                    # Process any pending events that weren't processed due to throttling
                    if event_handler.pending_events and (time.time() - event_handler.last_processed > interval):
                        event_handler.process_events()
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                self.stop_monitoring()
        
        self.is_monitoring = True
        self.monitor_thread = Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop monitoring the source directory."""
        if not self.is_monitoring:
            logger.warning("Monitoring is not active")
            return

        logger.info("Stopping directory monitoring")
        self.is_monitoring = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)
            self.monitor_thread = None
            
        logger.info("Directory monitoring stopped")

    def organize_duplicates(self) -> Dict[str, int]:
        """
        Find and organize duplicate files based on content.

        Returns:
            Dict[str, int]: Statistics about duplicates found and handled
        """
        stats = {'duplicates_found': 0, 'duplicates_moved': 0}
        dup_dir = os.path.join(self.target_dir, 'duplicates')
        os.makedirs(dup_dir, exist_ok=True)
        
        try:
            # First pass: Collect file hashes
            file_hashes = {}
            
            for root, _, files in os.walk(self.source_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    
                    # Skip files in the duplicates directory
                    if dup_dir in file_path:
                        continue
                    
                    # Calculate file hash (using a fast method for large files)
                    file_hash = self._calculate_file_hash(file_path)
                    
                    if file_hash in file_hashes:
                        # This is a duplicate
                        stats['duplicates_found'] += 1
                        original_path = file_hashes[file_hash]
                        
                        # Move the duplicate to the duplicates directory
                        target_path = os.path.join(dup_dir, filename)
                        if os.path.exists(target_path):
                            # Rename if target already exists
                            base, ext = os.path.splitext(filename)
                            new_name = f"{base}_{int(time.time())}{ext}"
                            target_path = os.path.join(dup_dir, new_name)
                        
                        # Create a text file with information about the original
                        info_path = f"{target_path}.info.txt"
                        with open(info_path, 'w') as f:
                            f.write(f"Original file: {original_path}\n")
                            f.write(f"Hash: {file_hash}\n")
                            f.write(f"Duplicate found: {time.ctime()}\n")
                        
                        shutil.move(file_path, target_path)
                        logger.info(f"Moved duplicate file {file_path} to {target_path}")
                        stats['duplicates_moved'] += 1
                    else:
                        # First occurrence of this file
                        file_hashes[file_hash] = file_path
            
            logger.info(f"Found {stats['duplicates_found']} duplicates, moved {stats['duplicates_moved']} files")
            return stats
            
        except Exception as e:
            logger.error(f"Error organizing duplicate files: {e}")
            raise

    def _calculate_file_hash(self, file_path: str, block_size: int = 65536) -> str:
        """
        Calculate a hash for a file using a fast method suitable for large files.
        
        Args:
            file_path (str): Path to the file
            block_size (int, optional): Size of blocks to read. Defaults to 65536.
            
        Returns:
            str: Hexadecimal hash string
        """
        import hashlib
        
        hasher = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                # Read the first and last blocks, plus some blocks from the middle for large files
                # This is faster than reading the entire file but still provides good collision resistance
                
                # Get file size
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                f.seek(0)  # Reset to beginning
                
                if file_size <= block_size * 3:
                    # For small files, just hash the entire content
                    buf = f.read()
                    hasher.update(buf)
                else:
                    # Read first block
                    buf = f.read(block_size)
                    hasher.update(buf)
                    
                    # Read a block from the middle
                    middle_pos = (file_size // 2) - (block_size // 2)
                    f.seek(middle_pos)
                    buf = f.read(block_size)
                    hasher.update(buf)
                    
                    # Read the last block
                    f.seek(-block_size, 2)
                    buf = f.read(block_size)
                    hasher.update(buf)
                
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return f"error_{int(time.time())}"  # Return a unique error hash to avoid false duplicates


@click.group()
def cli():
    """File Organizer - A tool to organize files by various criteria."""
    pass


@cli.command()
@click.argument('source_dir', type=click.Path(exists=True))
@click.option('--target-dir', '-t', type=click.Path(), help='Target directory. If not specified, uses source_dir.')
@click.option('--ignore', '-i', multiple=True, help='Extensions to ignore (without dot, e.g., "txt")')
def by_extension(source_dir, target_dir, ignore):
    """Organize files by their extensions."""
    organizer = FileOrganizer(source_dir, target_dir)
    result = organizer.organize_by_extension(ignore)
    click.echo(f"Organized {sum(result.values())} files by extension:")
    for ext, count in result.items():
        click.echo(f"  {ext}: {count} files")


@cli.command()
@click.argument('source_dir', type=click.Path(exists=True))
@click.option('--target-dir', '-t', type=click.Path(), help='Target directory. If not specified, uses source_dir.')
@click.option('--format', '-f', default='%Y-%m-%d', help='Date format for directories (default: YYYY-MM-DD)')
def by_date(source_dir, target_dir, format):
    """Organize files by their modification dates."""
    organizer = FileOrganizer(source_dir, target_dir)
    result = organizer.organize_by_date(format)
    click.echo(f"Organized {sum(result.values())} files by date:")
    for date_str, count in result.items():
        click.echo(f"  {date_str}: {count} files")


@cli.command()
@click.argument('source_dir', type=click.Path(exists=True))
@click.option('--target-dir', '-t', type=click.Path(), help='Target directory. If not specified, uses source_dir.')
def by_size(source_dir, target_dir):
    """Organize files by their sizes."""
    organizer = FileOrganizer(source_dir, target_dir)
    result = organizer.organize_by_size()
    click.echo(f"Organized {sum(result.values())} files by size:")
    for size_cat, count in result.items():
        click.echo(f"  {size_cat}: {count} files")


@cli.command()
@click.argument('source_dir', type=click.Path(exists=True))
@click.option('--target-dir', '-t', type=click.Path(), help='Target directory. If not specified, uses source_dir.')
def find_duplicates(source_dir, target_dir):
    """Find and organize duplicate files."""
    organizer = FileOrganizer(source_dir, target_dir)
    result = organizer.organize_duplicates()
    click.echo(f"Found {result['duplicates_found']} duplicate files")
    click.echo(f"Moved {result['duplicates_moved']} duplicate files to duplicates directory")


@cli.command()
@click.argument('source_dir', type=click.Path(exists=True))
@click.option('--target-dir', '-t', type=click.Path(), help='Target directory. If not specified, uses source_dir.')
@click.option('--method', '-m', type=click.Choice(['extension', 'date', 'size']), default='extension',
              help='Organization method to use when monitoring.')
@click.option('--interval', '-i', type=int, default=60, help='Monitoring interval in seconds (default: 60)')
def monitor(source_dir, target_dir, method, interval):
    """Monitor a directory and organize new files as they are added."""
    organizer = FileOrganizer(source_dir, target_dir)
    click.echo(f"Starting to monitor {source_dir} using {method} organization method...")
    click.echo("Press CTRL+C to stop monitoring.")
    try:
        organizer.start_monitoring(method, interval)
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping monitoring...")
        organizer.stop_monitoring()
        click.echo("Monitoring stopped.")


def main():
    """Main entry point for the file organizer CLI."""
    try:
        cli()
    except Exception as e:
        logger.error(f"Error: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
