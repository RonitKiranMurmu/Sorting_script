import os  # For operating system interactions (e.g., file paths)
import shutil  # For file operations like moving
import json  # For reading JSON config files
import logging  # For logging messages to file and console
import argparse  # For parsing command-line arguments
import csv  # For generating CSV reports
import hashlib  # For computing file hashes to detect duplicates
from pathlib import Path  # For cross-platform path handling
from typing import Dict, List, Optional, Tuple  # For type hints
from concurrent.futures import ThreadPoolExecutor  # For parallel file processing
try:
    from tqdm import tqdm  # Optional: for progress bar display
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False  # Flag to check if tqdm is available

# Define the FileOrganizer class to handle file organization
class FileOrganizer:
    # Default configuration for file categories and their extensions
    DEFAULT_CONFIG = {
        'Images': {'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'], 'folder': 'Images'},
        'Documents': {'extensions': ['.pdf', '.doc', '.docx', '.txt', '.rtf'], 'folder': 'Documents'},
        'Videos': {'extensions': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'], 'folder': 'Videos'},
        'Audio': {'extensions': ['.mp3', '.wav', '.flac', '.aac'], 'folder': 'Audio'},
        'Other': {'extensions': [], 'folder': 'Other'}  # Catch-all for uncategorized files
    }

    # Initialize the FileOrganizer with directory and options
    def __init__(self, directory: str, config_file: Optional[str] = None, 
                 dry_run: bool = False, recursive: bool = False, verbose: bool = False,
                 min_size: int = 0, quiet: bool = False, summary_only: bool = False,
                 max_workers: int = 4, report: bool = False, force: bool = False):
        self.directory = Path(directory).resolve()  # Resolve to absolute path
        self.dry_run = dry_run  # Simulate moves without executing
        self.recursive = recursive  # Process subdirectories if True
        self.moved_count = 0  # Counter for moved files
        self.skipped_count = 0  # Counter for skipped files
        self.duplicate_count = 0  # Counter for duplicate files
        self.min_file_size = min_size  # Minimum size to process files
        self.quiet = quiet  # Suppress output if True
        self.summary_only = summary_only  # Show only summary if True
        self.max_workers = max_workers  # Number of parallel workers
        self.report = report  # Generate CSV report if True
        self.force = force  # Overwrite existing files if True
        self.log_file = 'file_organizer.log'  # Log file name
        self.undo_file = 'undo_moves.sh' if os.name != 'nt' else 'undo_moves.bat'  # Undo script name (OS-specific)
        self.report_file = 'file_organizer_report.csv'  # Report file name
        self.undo_moves: List[Tuple[str, str]] = []  # List to store moves for undo
        self.file_hashes: Dict[str, Path] = {}  # Dictionary to store file hashes for duplicate detection

        # Set logging level based on quiet/verbose flags
        log_level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
            handlers=[
                logging.FileHandler(self.log_file),  # Log to file
                logging.StreamHandler() if not quiet else logging.NullHandler()  # Log to console unless quiet
            ]
        )
        self.logger = logging.getLogger(__name__)  # Get logger instance

        self.categories = self._load_config(config_file)  # Load category configuration

    # Load configuration from a JSON file or use default
    def _load_config(self, config_file: Optional[str]) -> Dict[str, dict]:
        """Load categories from a config file or use default if none provided."""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)  # Read JSON config
                self.logger.info(f"Loaded config from {config_file}")
                for key, value in config.items():
                    if 'folder' not in value:
                        value['folder'] = key  # Set folder name to category if not specified
                return config
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in {config_file}: {e}")
        return self.DEFAULT_CONFIG  # Fallback to default config

    # Create category folders in the target directory
    def _create_folders(self) -> None:
        """Create category folders if they don't exist."""
        for category in self.categories.values():
            folder_path = self.directory / category['folder']
            folder_path.mkdir(exist_ok=True)  # Create folder, ignore if it already exists

    # Determine the destination folder based on file extension
    def _get_destination(self, extension: str) -> Path:
        """Determine destination folder based on file extension."""
        extension = extension.lower()  # Case-insensitive matching
        for category, info in self.categories.items():
            if extension in info['extensions']:
                return self.directory / info['folder']  # Return category folder path
        return self.directory / self.categories['Other']['folder']  # Default to 'Other' folder

    # Compute SHA-256 hash of a file for duplicate detection
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):  # Read in 4KB chunks
                sha256.update(chunk)
        return sha256.hexdigest()  # Return hexadecimal hash

    # Generate a unique file path, handling duplicates and overwrites
    def _get_unique_path(self, dest_dir: Path, filename: str, file_hash: str) -> Optional[Path]:
        """Generate a unique file path, handling duplicates and overwrites."""
        dest_path = dest_dir / filename

        if file_hash in self.file_hashes:  # Check for duplicates
            self.duplicate_count += 1
            if not self.summary_only:
                self.logger.info(f"Skipped '{filename}' (duplicate of '{self.file_hashes[file_hash].name}')")
            return None  # Skip duplicate files

        if dest_path.exists():  # Check if file already exists at destination
            if self.force:
                return dest_path  # Overwrite if force is enabled
            else:
                base_name, ext = os.path.splitext(filename)
                counter = 1
                while True:  # Find a unique name by appending a number
                    new_name = f"{base_name}_{counter}{ext}"
                    new_path = dest_dir / new_name
                    if not new_path.exists():
                        return new_path
                    counter += 1
        return dest_path  # Return original path if no conflict

    # Process a single file (move it to the appropriate folder)
    def _process_file(self, file_path: Path) -> None:
        """Process a single file."""
        # Skip system files generated by this script
        if file_path.name in (self.log_file, self.undo_file, self.report_file):
            self.skipped_count += 1
            if not self.summary_only:
                self.logger.info(f"Skipped '{file_path.name}' (system file)")
            return

        # Skip if not a file (e.g., directories)
        if not file_path.is_file():
            self.skipped_count += 1
            if not self.summary_only:
                self.logger.info(f"Skipped '{file_path.name}' (not a file)")
            return

        # Skip hidden files (starting with '.')
        if file_path.name.startswith('.'):
            self.skipped_count += 1
            if not self.summary_only:
                self.logger.info(f"Skipped '{file_path.name}' (hidden file)")
            return

        # Check file size against minimum
        file_size = file_path.stat().st_size
        if file_size < self.min_file_size:
            self.skipped_count += 1
            if not self.summary_only:
                self.logger.info(f"Skipped '{file_path.name}' (size {file_size} < {self.min_file_size} bytes)")
            return

        try:
            extension = file_path.suffix  # Get file extension
            if not extension:
                self.skipped_count += 1
                if not self.summary_only:
                    self.logger.info(f"Skipped '{file_path.name}' (no extension)")
                return

            file_hash = self._compute_file_hash(file_path)  # Compute hash for duplicate check
            dest_dir = self._get_destination(extension)  # Get destination folder
            dest_path = self._get_unique_path(dest_dir, file_path.name, file_hash)  # Get unique destination path
            if dest_path is None:
                return  # Skip if no valid destination (e.g., duplicate)

            if self.dry_run:  # Simulate move in dry run mode
                if not self.summary_only:
                    self.logger.info(f"[DRY RUN] Would move '{file_path.name}' ({file_size} bytes) to '{dest_dir.name}'")
            else:  # Perform actual move
                shutil.move(str(file_path), str(dest_path))
                self.moved_count += 1
                self.undo_moves.append((str(dest_path), str(file_path)))  # Record move for undo
                self.file_hashes[file_hash] = dest_path  # Store hash
                if not self.summary_only:
                    self.logger.info(f"Moved '{file_path.name}' ({file_size} bytes) to '{dest_dir.name}'" +
                                   (f" (overwrote existing)" if dest_path.exists() and self.force else ""))

            if self.report:  # Add to CSV report if enabled
                self._add_to_report(file_path, dest_path, file_size, file_hash)

        except Exception as e:
            self.logger.error(f"Error processing '{file_path.name}': {str(e)}")
            self.skipped_count += 1

    # Add a file move to the CSV report
    def _add_to_report(self, src_path: Path, dest_path: Path, size: int, file_hash: str) -> None:
        """Add a move to the CSV report."""
        if not hasattr(self, '_report_writer'):  # Initialize report file if not already done
            self._report_file = open(self.report_file, 'w', newline='')
            self._report_writer = csv.writer(self._report_file)
            self._report_writer.writerow(['Source', 'Destination', 'Size (bytes)', 'SHA-256 Hash'])  # Header
        self._report_writer.writerow([str(src_path), str(dest_path), size, file_hash])  # Write row

    # Generate a script to undo all file moves
    def _generate_undo_script(self) -> None:
        """Generate a script to undo all moves."""
        if not self.undo_moves or self.dry_run:  # Skip if no moves or dry run
            return

        with open(self.undo_file, 'w') as f:
            if os.name == 'nt':  # Windows batch script
                f.write("@echo off\n")
                for dest, src in self.undo_moves:
                    f.write(f'move "{dest}" "{src}"\n')
                f.write("echo Undo complete!\n")
                f.write("pause\n")
            else:  # Unix shell script
                f.write("#!/bin/bash\n")
                for dest, src in self.undo_moves:
                    f.write(f'mv "{dest}" "{src}"\n')
                f.write("echo 'Undo complete!'\n")
        self.logger.info(f"Undo script generated: {self.undo_file}")

    # Main method to organize files in the directory
    def organize(self) -> None:
        """Organize files in the directory."""
        if not self.directory.exists():  # Check if directory exists
            self.logger.error(f"Directory '{self.directory}' does not exist")
            return

        self._create_folders()  # Create category folders
        if not self.quiet:
            self.logger.info(f"Starting organization in: {self.directory}")

        # Get list of files (recursive or not)
        files = list(self.directory.rglob('*') if self.recursive else self.directory.iterdir())
        if not files:
            self.logger.info("No files to process")
            return

        # Process files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            iterable = tqdm(files, desc="Processing files") if HAS_TQDM and not self.quiet else files
            executor.map(self._process_file, iterable)  # Map file processing to threads

        self._generate_undo_script()  # Generate undo script
        if self.report and not self.dry_run and hasattr(self, '_report_file'):  # Close report file
            self._report_file.close()
            self.logger.info(f"Report generated: {self.report_file}")
        self._print_summary()  # Print summary of operations

    # Print a summary of the organization process
    def _print_summary(self) -> None:
        """Print summary of operations."""
        if not self.quiet:
            self.logger.info("\nOrganization complete!")
            self.logger.info(f"Files moved: {self.moved_count}")
            self.logger.info(f"Files skipped: {self.skipped_count}")
            self.logger.info(f"Duplicates detected: {self.duplicate_count}")

# Parse command-line arguments
def parse_args():
    """Set up and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Organize files by extension with advanced options.")
    parser.add_argument("directory", nargs="?", default=os.getcwd(), 
                        help="Directory to organize (default: current directory)")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without moving files")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except errors")
    parser.add_argument("--summary-only", action="store_true", help="Show only the final summary")
    parser.add_argument("--min-size", type=int, default=0, 
                        help="Minimum file size in bytes to process (default: 0)")
    parser.add_argument("--max-workers", type=int, default=4, 
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--report", action="store_true", help="Generate a CSV report of moves")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files in destination")
    return parser.parse_args()

# Main entry point of the script
def main():
    """Initialize and run the file organizer with parsed arguments."""
    args = parse_args()  # Parse command-line arguments
    organizer = FileOrganizer(
        directory=args.directory,  # Use directory from command line (or current dir if not provided)
        config_file=args.config,
        dry_run=args.dry_run,
        recursive=args.recursive,
        verbose=args.verbose,
        min_size=args.min_size,
        quiet=args.quiet,
        summary_only=args.summary_only,
        max_workers=args.max_workers,
        report=args.report,
        force=args.force
    )
    organizer.organize()  # Start the organization process

# Run the script if executed directly
if __name__ == "__main__":
    main()