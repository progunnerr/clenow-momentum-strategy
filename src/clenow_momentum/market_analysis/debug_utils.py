"""Debug utilities for market data analysis.

This module handles debug file management and data saving for analysis purposes,
following the Single Responsibility Principle by focusing only on debugging concerns.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger


class DebugDataManager:
    """Manages debug data saving and cleanup for market analysis.

    Handles the creation, saving, and cleanup of debug files when
    DEBUG_MARKET_DATA environment variable is enabled.
    """

    def __init__(self, data_dir: Path | None = None, keep_files: int = 10):
        """Initialize debug data manager.

        Args:
            data_dir: Directory for debug files (default: data/debug)
            keep_files: Number of debug file sets to keep (default: 10)
        """
        self.data_dir = data_dir or Path("data/debug")
        self.keep_files = keep_files

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled via environment variable.

        Returns:
            True if DEBUG_MARKET_DATA is set to 'true', '1', or 'yes'
        """
        debug_env = os.getenv("DEBUG_MARKET_DATA", "").lower()
        return debug_env in ("true", "1", "yes")

    def save_debug_data(
        self, data: pd.DataFrame, filename: str, metadata: dict | None = None
    ) -> Path | None:
        """Save data for debugging purposes if debug mode is enabled.

        Args:
            data: DataFrame to save
            filename: Name of the file (without extension)
            metadata: Additional metadata to save with the data

        Returns:
            Path to saved file or None if debug mode is disabled
        """
        if not self.is_debug_enabled():
            return None

        if data is None or data.empty:
            logger.debug(f"Skipping save of empty data: {filename}")
            return None

        # Create directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for file naming
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Save as CSV for easy inspection
        csv_path = self.data_dir / f"{timestamp}_{filename}.csv"
        try:
            data.to_csv(csv_path)
            logger.debug(f"Saved debug data to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug CSV {csv_path}: {e}")
            return None

        # Save metadata as JSON
        self._save_metadata(data, timestamp, filename, metadata)

        # Clean up old debug files
        self._cleanup_old_files()

        return csv_path

    def _save_metadata(
        self,
        data: pd.DataFrame,
        timestamp: str,
        filename: str,
        additional_metadata: dict | None = None,
    ) -> None:
        """Save metadata about the debug data.

        Args:
            data: DataFrame that was saved
            timestamp: Timestamp used in filename
            filename: Base filename
            additional_metadata: Additional metadata to include
        """
        metadata = {
            "timestamp": timestamp,
            "datetime": datetime.now(UTC).isoformat(),
            "shape": list(data.shape),
            "columns": (
                list(data.columns)
                if not isinstance(data.columns, pd.MultiIndex)
                else str(data.columns)
            ),
            "index_type": str(type(data.index)),
            "dtypes": {str(col): str(dtype) for col, dtype in data.dtypes.items()},
            "has_multiindex_columns": isinstance(data.columns, pd.MultiIndex),
        }

        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        json_path = self.data_dir / f"{timestamp}_{filename}_metadata.json"
        try:
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Saved metadata to {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata {json_path}: {e}")

    def _cleanup_old_files(self) -> None:
        """Clean up old debug files, keeping only the most recent ones.

        Removes older CSV and JSON files based on the keep_files setting.
        """
        try:
            # Get all CSV files, sorted by name (which includes timestamp)
            csv_files = sorted(self.data_dir.glob("*.csv"), reverse=True)

            # Keep only the most recent files
            if len(csv_files) > self.keep_files:
                files_to_remove = csv_files[self.keep_files :]

                for old_file in files_to_remove:
                    # Remove CSV file
                    old_file.unlink()

                    # Remove corresponding JSON files
                    json_file = old_file.with_suffix(".json")
                    if json_file.exists():
                        json_file.unlink()

                    # Remove metadata files
                    metadata_file = self.data_dir / f"{old_file.stem}_metadata.json"
                    if metadata_file.exists():
                        metadata_file.unlink()

                    logger.debug(f"Removed old debug file: {old_file.name}")

        except Exception as e:
            logger.warning(f"Error cleaning up old debug files: {e}")

    def get_debug_files(self) -> list[Path]:
        """Get list of current debug files.

        Returns:
            List of debug CSV file paths, sorted by timestamp (newest first)
        """
        if not self.data_dir.exists():
            return []

        return sorted(self.data_dir.glob("*.csv"), reverse=True)

    def clear_all_debug_files(self) -> int:
        """Remove all debug files.

        Returns:
            Number of files removed
        """
        if not self.data_dir.exists():
            return 0

        files_removed = 0
        try:
            for file_path in self.data_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    files_removed += 1

            logger.info(f"Removed {files_removed} debug files from {self.data_dir}")

        except Exception as e:
            logger.error(f"Error clearing debug files: {e}")

        return files_removed
