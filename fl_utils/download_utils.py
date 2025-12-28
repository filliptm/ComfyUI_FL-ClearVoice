"""
Download utilities for FL-ClearVoice with improved progress bars.

Provides a unified download experience with:
- Real-time progress bars that update in-place
- ETA and download speed display
- File-by-file progress tracking
"""

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass


def get_terminal_width() -> int:
    """Get terminal width, with fallback."""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


@dataclass
class DownloadProgress:
    """Track download progress for a single file."""
    filename: str
    total_size: int
    downloaded: int = 0
    start_time: float = 0.0

    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()

    @property
    def progress(self) -> float:
        """Return progress as 0.0-1.0."""
        if self.total_size <= 0:
            return 0.0
        return min(self.downloaded / self.total_size, 1.0)

    @property
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def speed(self) -> float:
        """Return download speed in bytes/second."""
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0.0
        return self.downloaded / elapsed

    @property
    def eta(self) -> float:
        """Return estimated time remaining in seconds."""
        speed = self.speed
        if speed <= 0:
            return 0.0
        remaining = self.total_size - self.downloaded
        return remaining / speed


class ProgressBar:
    """
    Console progress bar that updates in-place.

    Usage:
        pbar = ProgressBar(prefix="[FL ClearVoice]")
        pbar.start("Downloading model.bin", total_size=1000000)
        pbar.update(50000)
        pbar.update(100000)
        pbar.finish()
    """

    def __init__(self, prefix: str = "[FL ClearVoice]", bar_width: int = 30, update_interval: float = 0.1):
        self.prefix = prefix
        self.bar_width = bar_width
        self.update_interval = update_interval  # Minimum seconds between updates
        self.current: Optional[DownloadProgress] = None
        self._last_line_len = 0
        self._last_update_time = 0.0
        # Always try to use in-place updates - most modern terminals support \r
        self._use_inplace = True

    def _clear_line(self):
        """Clear the current line using carriage return."""
        # Use \r to return to start of line, then spaces to clear, then \r again
        sys.stdout.write('\r' + ' ' * self._last_line_len + '\r')
        sys.stdout.flush()

    def _write_progress(self, force: bool = False):
        """Write the current progress bar."""
        if self.current is None:
            return

        # Throttle updates to avoid too much output
        current_time = time.time()
        if not force and (current_time - self._last_update_time) < self.update_interval:
            return
        self._last_update_time = current_time

        p = self.current
        pct = int(p.progress * 100)
        filled = int(p.progress * self.bar_width)
        bar = '█' * filled + '░' * (self.bar_width - filled)

        # Format sizes
        downloaded_str = format_size(p.downloaded)
        total_str = format_size(p.total_size)
        speed_str = format_size(int(p.speed)) + "/s"

        # Format ETA
        if p.progress < 1.0 and p.speed > 0:
            eta_str = format_time(p.eta)
        else:
            eta_str = "--"

        # Truncate filename if needed
        max_filename_len = 25
        filename = p.filename
        if len(filename) > max_filename_len:
            filename = filename[:max_filename_len-3] + "..."

        # Build the line
        line = f"{self.prefix} {filename}: |{bar}| {pct}% ({downloaded_str}/{total_str}) {speed_str} ETA: {eta_str}"

        # Always use carriage return for in-place update
        # This works in most terminals including ComfyUI's output
        sys.stdout.write('\r' + line)
        sys.stdout.flush()
        self._last_line_len = len(line)

    def start(self, filename: str, total_size: int):
        """Start tracking a new file download."""
        self.current = DownloadProgress(
            filename=filename,
            total_size=total_size,
            downloaded=0,
            start_time=time.time()
        )
        self._last_update_time = 0.0  # Force first update
        self._write_progress(force=True)

    def update(self, downloaded: int):
        """Update the downloaded byte count (absolute, not delta)."""
        if self.current is not None:
            self.current.downloaded = downloaded
            self._write_progress()

    def update_delta(self, delta: int):
        """Update by adding delta bytes to current count."""
        if self.current is not None:
            self.current.downloaded += delta
            self._write_progress()

    def finish(self, message: str = None):
        """Finish the current download and print completion message."""
        if self.current is not None:
            p = self.current
            elapsed_str = format_time(p.elapsed)
            total_str = format_size(p.total_size)

            # Clear the progress line and print final message on new line
            self._clear_line()

            if message:
                print(f"{self.prefix} {message}")
            else:
                print(f"{self.prefix} Downloaded {p.filename} ({total_str}) in {elapsed_str}")

        self.current = None
        self._last_line_len = 0

    def print(self, message: str):
        """Print a message, preserving the progress bar."""
        self._clear_line()
        print(message)
        if self.current is not None:
            self._write_progress(force=True)


class MultiFileDownloader:
    """
    Download multiple files with overall progress tracking.

    Usage:
        downloader = MultiFileDownloader(prefix="[FL ClearVoice]")
        downloader.download_files(repo_id, files, local_dir)
    """

    def __init__(self, prefix: str = "[FL ClearVoice]"):
        self.prefix = prefix
        self.pbar = ProgressBar(prefix=prefix)

    def download_hf_files(
        self,
        repo_id: str,
        filenames: List[str],
        local_dir: Path,
        revision: str = "main"
    ) -> Path:
        """
        Download files from HuggingFace Hub with progress bars.

        Args:
            repo_id: HuggingFace repository ID
            filenames: List of filenames to download
            local_dir: Local directory to save files
            revision: Git revision (branch, tag, commit)

        Returns:
            Path to local directory
        """
        try:
            from huggingface_hub import HfApi, hf_hub_url
            import requests
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Please install with: pip install huggingface_hub"
            )

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        api = HfApi()

        # Get file info for sizes
        print(f"{self.prefix} Fetching file info from {repo_id}...")
        try:
            repo_info = api.repo_info(repo_id=repo_id, revision=revision, files_metadata=True)
            file_sizes = {}
            if hasattr(repo_info, 'siblings') and repo_info.siblings:
                for sibling in repo_info.siblings:
                    if hasattr(sibling, 'size') and sibling.size:
                        file_sizes[sibling.rfilename] = sibling.size
        except Exception as e:
            print(f"{self.prefix} Could not fetch file sizes: {e}")
            file_sizes = {}

        total_files = len(filenames)

        for idx, filename in enumerate(filenames, 1):
            local_path = local_dir / filename

            # Skip if already exists
            if local_path.exists():
                print(f"{self.prefix} [{idx}/{total_files}] {filename} (already exists)")
                continue

            # Get file size
            file_size = file_sizes.get(filename, 0)

            print(f"{self.prefix} [{idx}/{total_files}] Downloading {filename}...")

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Get the download URL
                url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)

                # Download with streaming for real-time progress
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()

                # Get actual file size from headers if not known
                if file_size == 0:
                    file_size = int(response.headers.get('content-length', 0))

                if file_size > 0:
                    self.pbar.start(filename, file_size)

                # Download in chunks with progress updates
                downloaded = 0
                chunk_size = 8192  # 8KB chunks

                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if file_size > 0:
                                self.pbar.update(downloaded)

                if file_size > 0:
                    self.pbar.finish()
                else:
                    actual_size = local_path.stat().st_size if local_path.exists() else 0
                    print(f"{self.prefix} Downloaded {filename} ({format_size(actual_size)})")

            except Exception as e:
                if file_size > 0:
                    self.pbar.finish(f"Error downloading {filename}: {e}")
                else:
                    print(f"{self.prefix} Error downloading {filename}: {e}")
                # Clean up partial download
                if local_path.exists():
                    local_path.unlink()
                raise

        print(f"{self.prefix} All files downloaded to {local_dir}")
        return local_dir

    def download_url(
        self,
        url: str,
        local_path: Path,
        description: str = None
    ) -> Path:
        """
        Download a file from URL with progress bar.

        Args:
            url: URL to download from
            local_path: Local path to save file
            description: Optional description for progress bar

        Returns:
            Path to downloaded file
        """
        import urllib.request

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        filename = description or local_path.name

        # Get file size from headers
        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('content-length', 0))
        except Exception:
            total_size = 0

        if total_size > 0:
            self.pbar.start(filename, total_size)
        else:
            print(f"{self.prefix} Downloading {filename}...")

        # Download with progress
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                self.pbar.update(min(downloaded, total_size))

        try:
            urllib.request.urlretrieve(url, str(local_path), reporthook=reporthook)

            if total_size > 0:
                self.pbar.finish()
            else:
                actual_size = local_path.stat().st_size if local_path.exists() else 0
                print(f"{self.prefix} Downloaded {filename} ({format_size(actual_size)})")

            return local_path

        except Exception as e:
            if total_size > 0:
                self.pbar.finish(f"Error downloading {filename}: {e}")
            else:
                print(f"{self.prefix} Error downloading {filename}: {e}")
            raise


def download_with_progress(
    repo_id: str,
    filenames: List[str],
    local_dir: Path,
    prefix: str = "[FL ClearVoice]"
) -> Path:
    """
    Convenience function to download HuggingFace files with progress.

    Args:
        repo_id: HuggingFace repository ID
        filenames: List of filenames to download
        local_dir: Local directory to save files
        prefix: Prefix for progress messages

    Returns:
        Path to local directory
    """
    downloader = MultiFileDownloader(prefix=prefix)
    return downloader.download_hf_files(repo_id, filenames, local_dir)


def download_url_with_progress(
    url: str,
    local_path: Path,
    description: str = None,
    prefix: str = "[FL ClearVoice]"
) -> Path:
    """
    Convenience function to download URL with progress.

    Args:
        url: URL to download from
        local_path: Local path to save file
        description: Optional description for progress bar
        prefix: Prefix for progress messages

    Returns:
        Path to downloaded file
    """
    downloader = MultiFileDownloader(prefix=prefix)
    return downloader.download_url(url, local_path, description)
