#!/usr/bin/env python3
"""
VASP OUTCAR Analysis Tool
Parses VASP OUTCAR files to identify and classify errors and warnings.

This tool discovers VASP job directories, parses their OUTCAR files for errors
and warnings, classifies them, and generates comprehensive reports.
"""

import argparse
import datetime
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Issue:
    """Represents a single error or warning occurrence."""

    type: Literal["error", "warning"]
    pattern: str  # The actual pattern text matched
    context: str  # Full context (pattern + surrounding lines)
    line_number: int = 0  # Where it occurred in the file


@dataclass
class VaspJob:
    """Represents a single VASP calculation directory."""

    path: Path
    name: str
    errors: List[Issue] = field(default_factory=list)
    warnings: List[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def is_clean(self) -> bool:
        return not (self.has_errors or self.has_warnings)

    @property
    def all_issues(self) -> List[Issue]:
        """Return all issues (errors first, then warnings)."""
        return self.errors + self.warnings


@dataclass
class IssueGroup:
    """A group of jobs with the same issue."""

    issue_id: str  # e.g., "er1", "warn3"
    issue_type: Literal["error", "warning"]
    pattern: str  # The normalized/deduplicated pattern
    context: str  # Representative context
    jobs: List[VaspJob] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.jobs)


@dataclass
class AnalysisResults:
    """Container for complete analysis results."""

    jobs: List[VaspJob]
    groups: List[IssueGroup]

    @property
    def total_jobs(self) -> int:
        return len(self.jobs)

    @property
    def jobs_with_errors(self) -> int:
        return sum(1 for job in self.jobs if job.has_errors)

    @property
    def jobs_with_warnings(self) -> int:
        return sum(1 for job in self.jobs if job.has_warnings and not job.has_errors)

    @property
    def clean_jobs(self) -> int:
        return sum(1 for job in self.jobs if job.is_clean)


# ============================================================================
# Core Components
# ============================================================================


class OutcarParser:
    """Parses OUTCAR files to extract all errors and warnings."""

    ERROR_PATTERN = "EEEEEEE  RRRRRR   RRRRRR"
    WARNING_PATTERN = "RRRRR   N    N  II  N    N   GGGG"
    CONTEXT_LINES = 11

    IGNORED_WARNING_PATTERNS = ["For optimal performance we recommend to set"]

    def parse_file(self, outcar_path: Path) -> Tuple[List[Issue], List[Issue]]:
        """
        Parse a single OUTCAR file.

        Args:
            outcar_path: Path to OUTCAR file

        Returns:
            Tuple of (all_errors, all_warnings)
        """
        if not outcar_path.exists():
            return [], []

        errors = self._extract_issues_ripgrep(outcar_path, self.ERROR_PATTERN, "error")
        warnings = self._extract_issues_ripgrep(
            outcar_path, self.WARNING_PATTERN, "warning"
        )

        return errors, warnings

    def _extract_issues_ripgrep(
        self, file_path: Path, pattern: str, issue_type: Literal["error", "warning"]
    ) -> List[Issue]:
        """
        Extract all occurrences of a pattern using ripgrep.

        Args:
            file_path: Path to file to search
            pattern: Pattern to search for
            issue_type: Type of issue ("error" or "warning")

        Returns:
            List of Issue objects
        """
        try:
            result = subprocess.run(
                ["rg", pattern, "-A", str(self.CONTEXT_LINES), str(file_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            if not result.stdout:
                return []

            # Split by '--' separator to get individual matches
            sections = result.stdout.split("--")
            issues = []

            for section in sections:
                section = section.strip()
                if section:
                    issues.append(
                        Issue(
                            type=issue_type,
                            pattern=pattern,
                            context=section,
                            line_number=0,  # Could extract from ripgrep if needed
                        )
                    )

            # Add filtering at the end before return
            if issue_type == "warning":
                filtered_issues = []
                for issue in issues:
                    if self._should_ignore(issue):
                        logging.debug(f"Ignoring warning: {issue.context[-73:-13]}...")
                    else:
                        filtered_issues.append(issue)
                issues = filtered_issues

            return issues

        except subprocess.SubprocessError as e:
            logging.info(f"Error running ripgrep on {file_path}: {e}")
            return []

    def _should_ignore(self, issue: Issue) -> bool:
        """Check if issue should be ignored."""
        for ignored_pattern in self.IGNORED_WARNING_PATTERNS:
            if ignored_pattern in issue.context:
                return True
        return False


class JobDiscovery:
    """Discovers VASP job directories."""

    def __init__(self, base_dir: Path, pattern: str):
        self.base_dir = base_dir
        self.pattern = pattern

    def find_jobs(self) -> List[Path]:
        """
        Find all directories matching the pattern with OUTCAR files.

        Returns:
            List of valid VASP job directory paths
        """
        job_paths = []

        for path in self.base_dir.glob(self.pattern):
            if self.validate_job(path):
                job_paths.append(path)

        return sorted(job_paths)  # Sort for consistent ordering

    def validate_job(self, path: Path) -> bool:
        """
        Check if a directory is a valid VASP job.

        Args:
            path: Directory path to validate

        Returns:
            True if valid VASP job directory
        """
        return path.is_dir() and (path / "OUTCAR").exists()


class JobAnalyzer:
    """Analyzes VASP jobs to extract issues."""

    def __init__(self, parser: OutcarParser):
        self.parser = parser

    def analyze_job(self, job_path: Path) -> VaspJob:
        """
        Analyze a single job directory.

        Args:
            job_path: Path to VASP job directory

        Returns:
            VaspJob object with all extracted issues
        """
        errors, warnings = self.parser.parse_file(job_path / "OUTCAR")

        return VaspJob(
            path=job_path, name=job_path.name, errors=errors, warnings=warnings
        )

    def analyze_batch(
        self,
        job_paths: List[Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[VaspJob]:
        """
        Analyze multiple jobs with optional progress reporting.

        Args:
            job_paths: List of job directory paths
            progress_callback: Optional callback(current, total, name)

        Returns:
            List of VaspJob objects
        """
        jobs = []
        total = len(job_paths)

        for i, path in enumerate(job_paths):
            if progress_callback and (i % 10 == 0 or i == total - 1):
                progress_callback(i + 1, total, path.name)

            jobs.append(self.analyze_job(path))

        return jobs


class IssueClassifier:
    """Classifies and groups issues across jobs."""

    def classify(self, jobs: List[VaspJob]) -> List[IssueGroup]:
        """
        Group jobs by unique issue patterns.

        Jobs with multiple issues will appear in multiple groups.
        Groups are sorted by frequency (most common first).

        Args:
            jobs: List of analyzed VaspJob objects

        Returns:
            List of IssueGroup objects
        """
        # Collect all unique issues
        error_patterns: Dict[str, List[Tuple[VaspJob, Issue]]] = defaultdict(list)
        warning_patterns: Dict[str, List[Tuple[VaspJob, Issue]]] = defaultdict(list)

        for job in jobs:
            # Group by error contexts
            for error in job.errors:
                error_patterns[error.context].append((job, error))

            # Group by warning contexts
            for warning in job.warnings:
                warning_patterns[warning.context].append((job, warning))

        # Create issue groups
        groups = []

        # Process errors first
        error_groups = self._create_groups(error_patterns, "error")
        groups.extend(error_groups)

        # Then process warnings
        warning_groups = self._create_groups(warning_patterns, "warning")
        groups.extend(warning_groups)

        # Assign IDs based on frequency
        self._assign_ids(groups)

        return groups

    def _create_groups(
        self,
        pattern_map: Dict[str, List[Tuple[VaspJob, Issue]]],
        issue_type: Literal["error", "warning"],
    ) -> List[IssueGroup]:
        """Create IssueGroup objects from pattern map."""
        groups = []

        for context, job_issue_pairs in pattern_map.items():
            # Extract unique jobs (a job might have same issue multiple times)
            unique_jobs_dict = {}
            for job, _ in job_issue_pairs:
                unique_jobs_dict[job.path] = job  # or job.name if you prefer
            unique_jobs = list(unique_jobs_dict.values())

            # Use the first issue's pattern as representative
            representative_issue = job_issue_pairs[0][1]

            group = IssueGroup(
                issue_id="",  # Will be assigned later
                issue_type=issue_type,
                pattern=representative_issue.pattern,
                context=context,
                jobs=unique_jobs,
            )
            groups.append(group)

        return groups

    def _assign_ids(self, groups: List[IssueGroup]) -> None:
        """
        Assign IDs based on frequency and type.

        Errors get er1, er2, ... (sorted by frequency)
        Warnings get warn1, warn2, ... (sorted by frequency)
        """
        # Separate and sort by frequency
        error_groups = [g for g in groups if g.issue_type == "error"]
        warning_groups = [g for g in groups if g.issue_type == "warning"]

        error_groups.sort(key=lambda g: g.count, reverse=True)
        warning_groups.sort(key=lambda g: g.count, reverse=True)

        # Assign IDs
        for i, group in enumerate(error_groups, 1):
            group.issue_id = f"er{i}"

        for i, group in enumerate(warning_groups, 1):
            group.issue_id = f"warn{i}"


class OutputManager:
    """Manages all output generation."""

    def __init__(self, base_dir: Path, output_subdir: str = "x-py-bug-groups"):
        self.base_dir = base_dir
        self.output_dir = base_dir / output_subdir

    def write_bad_vasprun(self, job: VaspJob) -> None:
        """
        Write BAD_VASPRUN file in the job directory.

        Args:
            job: VaspJob object with issues to write
        """
        bad_vasprun_path = job.path / "BAD_VASPRUN"

        with open(bad_vasprun_path, "w") as f:
            if job.errors:
                f.write("ERRORS:\n")
                f.write("=" * 80 + "\n\n")
                for i, error in enumerate(job.errors, 1):
                    f.write(f"Error {i}:\n")
                    f.write(error.context)
                    f.write("\n\n")

            if job.warnings:
                f.write("WARNINGS:\n")
                f.write("=" * 80 + "\n\n")
                for i, warning in enumerate(job.warnings, 1):
                    f.write(f"Warning {i}:\n")
                    f.write(warning.context)
                    f.write("\n\n")

    def write_summary(
        self, results: AnalysisResults, action: Optional[str] = None
    ) -> Path:
        """
        Write comprehensive summary file.

        Args:
            results: AnalysisResults object
            action: Optional action performed (copy/move/symlink)

        Returns:
            Path to summary file
        """
        self.output_dir.mkdir(exist_ok=True)
        summary_path = self.output_dir / "summary.txt"

        with open(summary_path, "w") as f:
            self._write_header(f, results)
            self._write_statistics(f, results)
            self._write_group_summary(f, results)
            self._write_detailed_groups(f, results)
            self._write_all_dirs(f, results)

            if action:
                f.write(f"\nAction '{action}' was performed for all directories\n")

            f.write(f"\nEnd of summary\n")

        self.write_unique_dirs_to_separate_files(results)

        return summary_path

    def _write_header(self, f, results: AnalysisResults) -> None:
        """Write summary header."""
        f.write("VASP OUTCAR Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Base Directory: {self.base_dir}\n\n")

    def _write_statistics(self, f, results: AnalysisResults) -> None:
        """Write statistics section."""
        f.write("Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total VASP directories processed: {results.total_jobs}\n")
        f.write(
            f"Directories with errors: {results.jobs_with_errors} "
            f"({results.jobs_with_errors/results.total_jobs*100:.1f}%)\n"
        )
        f.write(
            f"Directories with warnings only: {results.jobs_with_warnings} "
            f"({results.jobs_with_warnings/results.total_jobs*100:.1f}%)\n"
        )
        f.write(
            f"Clean directories: {results.clean_jobs} "
            f"({results.clean_jobs/results.total_jobs*100:.1f}%)\n"
        )

        error_groups = [g for g in results.groups if g.issue_type == "error"]
        warning_groups = [g for g in results.groups if g.issue_type == "warning"]

        f.write(f"Unique error types: {len(error_groups)}\n")
        f.write(f"Unique warning types: {len(warning_groups)}\n\n")

    def _write_group_summary(self, f, results: AnalysisResults) -> None:
        """Write bug groups summary."""
        f.write("Bug Groups Summary:\n")
        f.write("-" * 80 + "\n")

        # Sort: errors first, then warnings, both by frequency
        sorted_groups = sorted(
            results.groups,
            key=lambda g: (0 if g.issue_type == "error" else 1, -g.count),
        )

        for group in sorted_groups:
            percent = group.count / results.total_jobs * 100
            f.write(f"{group.issue_id}: {group.count} occurrences ({percent:.1f}%)\n")

        f.write("\n")

    def _write_detailed_groups(self, f, results: AnalysisResults) -> None:
        """Write detailed information for each group."""
        f.write("Detailed Bug Groups:\n")
        f.write("-" * 80 + "\n")

        error_groups = [g for g in results.groups if g.issue_type == "error"]
        warning_groups = [g for g in results.groups if g.issue_type == "warning"]

        # Sort by frequency
        error_groups.sort(key=lambda g: g.count, reverse=True)
        warning_groups.sort(key=lambda g: g.count, reverse=True)

        if error_groups:
            f.write("\nError Groups:\n")
            for group in error_groups:
                self._write_group_detail(f, group)

        if warning_groups:
            f.write("\nWarning Groups:\n")
            for group in warning_groups:
                self._write_group_detail(f, group)

    def _write_group_detail(self, f, group: IssueGroup, chars: int = 1400) -> None:
        """Write details for a single group."""
        f.write(f"\n{group.issue_id} ({group.count} occurrences):\n")
        f.write(f"{group.issue_type.capitalize()} excerpt (first {chars} chars):\n")
        f.write(f"{group.context[:chars]}...\n")
        f.write(f"Affected directories:\n")
        for job in group.jobs:
            f.write(f"  {job.path.relative_to(self.base_dir)}\n")

    def _write_all_dirs(self, f, results: AnalysisResults) -> None:
        """Write all unique directories with warnings and errors."""
        f.write("\nAll Unique Directories:\n")
        f.write("=" * 80 + "\n")

        # Collect unique directories (sets handle deduplication automatically!)
        error_dirs = {job.path for job in results.jobs if job.has_errors}
        warning_dirs = {job.path for job in results.jobs if job.has_warnings}

        # Write error directories
        f.write(f"\nUnique Directories with Errors ({len(error_dirs)} total):\n")
        f.write("-" * 80 + "\n")
        for path in sorted(error_dirs):
            f.write(f"{path.relative_to(self.base_dir)}\n")

        # Write warning directories
        f.write(f"\nUnique Directories with Warnings ({len(warning_dirs)} total):\n")
        f.write("-" * 80 + "\n")
        for path in sorted(warning_dirs):
            f.write(f"{path.relative_to(self.base_dir)}\n")

        f.write("\n")

    def write_unique_dirs_to_separate_files(self, results: AnalysisResults) -> None:
        """Write directories to error-dirs.txt and warning-dirs.txt"""
        self.output_dir.mkdir(exist_ok=True)

        error_dirs = {job.path for job in results.jobs if job.has_errors}
        warning_dirs = {job.path for job in results.jobs if job.has_warnings}

        # Write error-dirs.txt
        with open(self.output_dir / "error-dirs.txt", "w") as f:
            # f.write(f"# Directories with errors ({len(error_dirs)} total)\n")
            for path in sorted(error_dirs):
                f.write(f"{path.relative_to(self.base_dir)}\n")

        # Write warning-dirs.txt
        with open(self.output_dir / "warning-dirs.txt", "w") as f:
            # f.write(f"# Directories with warnings ({len(warning_dirs)} total)\n")
            for path in sorted(warning_dirs):
                f.write(f"{path.relative_to(self.base_dir)}\n")

    def create_group_structure(
        self,
        groups: List[IssueGroup],
        action: Optional[str] = None,
        fs_ops: Optional["FileSystemOperations"] = None,
    ) -> None:
        """
        Create bug group directory structure.

        Args:
            groups: List of IssueGroup objects
            action: Optional action to perform (copy/move/symlink)
            fs_ops: FileSystemOperations instance for performing actions
        """
        self.output_dir.mkdir(exist_ok=True)

        for i, group in enumerate(groups, 1):
            bug_dir = self.output_dir / group.issue_id
            bug_dir.mkdir(exist_ok=True)

            # Write error/warning text
            with open(bug_dir / "error_text.txt", "w") as f:
                f.write(group.context)

            # Write directories list
            with open(bug_dir / "directories.txt", "w") as f:
                for job in group.jobs:
                    f.write(f"{job.path.relative_to(self.base_dir)}\n")

            # Perform file operations if requested
            if action and fs_ops:
                total_groups = len(groups)
                logging.info(
                    f"Processing {group.issue_id} ({i}/{total_groups}) - "
                    f"{group.count} directories"
                )

                dirs_dir = bug_dir / "dirs"
                dirs_dir.mkdir(exist_ok=True)

                for j, job in enumerate(group.jobs):
                    if j % 5 == 0 or j == len(group.jobs) - 1:
                        logging.info(
                            f"  Processing directory {j+1}/{len(group.jobs)} "
                            f"({(j+1)/len(group.jobs)*100:.1f}%)"
                        )

                    target_dir = dirs_dir / job.name
                    if not target_dir.exists():
                        fs_ops.perform_action(action, job.path, target_dir)


class FileSystemOperations:
    """Handles file system operations (copy/move/symlink)."""

    @staticmethod
    def copy_directory(source: Path, target: Path) -> None:
        """Copy directory with error handling."""
        try:
            shutil.copytree(source, target)
        except Exception as e:
            logging.info(f"Error copying {source} to {target}: {e}")

    @staticmethod
    def move_directory(source: Path, target: Path) -> None:
        """Move directory with error handling."""
        try:
            shutil.move(str(source), str(target))
        except Exception as e:
            logging.info(f"Error moving {source} to {target}: {e}")

    @staticmethod
    def create_symlink(source: Path, target: Path) -> None:
        """Create symbolic link with error handling."""
        try:
            target.symlink_to(source.resolve(), target_is_directory=True)
        except Exception as e:
            logging.info(f"Error creating symlink {source} -> {target}: {e}")

    def perform_action(self, action: str, source: Path, target: Path) -> None:
        """
        Dispatch to appropriate method based on action.

        Args:
            action: One of 'copy', 'move', or 'symlink'
            source: Source directory path
            target: Target directory path
        """
        if action == "copy":
            self.copy_directory(source, target)
        elif action == "move":
            self.move_directory(source, target)
        elif action == "symlink":
            self.create_symlink(source, target)
        else:
            raise ValueError(f"Unknown action: {action}")


# ============================================================================
# Main Pipeline
# ============================================================================


class VaspAnalysisPipeline:
    """Main orchestrator for VASP analysis."""

    def __init__(self, base_dir: Path, pattern="*", action: Optional[str] = None):
        self.base_dir = base_dir
        self.action = action

        # Initialize components
        self.discovery = JobDiscovery(base_dir, pattern=pattern)
        self.parser = OutcarParser()
        self.analyzer = JobAnalyzer(self.parser)
        self.classifier = IssueClassifier()
        self.output = OutputManager(base_dir)
        self.fs_ops = FileSystemOperations()

    def run(self) -> AnalysisResults:
        """
        Execute the complete analysis pipeline.

        Returns:
            AnalysisResults object with all jobs and groups
        """
        logging.info(f"Parsing OUTCARS in {self.base_dir}...")

        # 1. Discover
        logging.info("")
        logging.info("## Step 1: Discovering VASP directories...")
        job_paths = self.discovery.find_jobs()
        logging.info(f"Found {len(job_paths)} VASP directories to process")

        if not job_paths:
            logging.info("No VASP directories found!")
            return AnalysisResults(jobs=[], groups=[])

        # 2. Analyze
        logging.info("")
        logging.info("## Step 2: Analyzing OUTCAR files...")
        jobs = self.analyzer.analyze_batch(job_paths, self._progress_callback)

        # 3. Write BAD_VASPRUN files
        logging.info("")
        logging.info("## Step 3: Writing BAD_VASPRUN files...")
        bad_count = 0
        for job in jobs:
            if not job.is_clean:
                self.output.write_bad_vasprun(job)
                bad_count += 1
        logging.info(f"Wrote {bad_count} BAD_VASPRUN files")

        # 4. Classify
        logging.info("")
        logging.info("## Step 4: Classifying issues...")
        groups = self.classifier.classify(jobs)
        logging.info(f"Created {len(groups)} issue groups")

        # Create results object
        results = AnalysisResults(jobs=jobs, groups=groups)

        # 5. Output
        logging.info("")
        logging.info("## Step 5: Generating outputs...")
        summary_path = self.output.write_summary(results, self.action)
        logging.info(f"Summary written to {summary_path}")

        logging.info("")
        logging.info("## Step 6: Creating bug group directories...")
        self.output.create_group_structure(groups, self.action, self.fs_ops)

        # 7. Report
        self._report_stats(results)

        return results

    def _progress_callback(self, current: int, total: int, name: str) -> None:
        """Progress callback for batch analysis."""
        logging.info(
            f"Processing directory {current}/{total} "
            f"({current/total*100:.1f}%) - {name}"
        )

    def _report_stats(self, results: AnalysisResults) -> None:
        """Print summary statistics."""
        logging.info("=" * 80)
        logging.info("ANALYSIS COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Total directories: {results.total_jobs}")
        logging.info(f"Directories with errors: {results.jobs_with_errors}")
        logging.info(f"Directories with warnings only: {results.jobs_with_warnings}")
        logging.info(f"Clean directories: {results.clean_jobs}")

        error_groups = [g for g in results.groups if g.issue_type == "error"]
        warning_groups = [g for g in results.groups if g.issue_type == "warning"]

        logging.info(f"Unique error types: {len(error_groups)}")
        logging.info(f"Unique warning types: {len(warning_groups)}")

        logging.info("Bug group summary:")
        # Sort by type and frequency
        sorted_groups = sorted(
            results.groups,
            key=lambda g: (0 if g.issue_type == "error" else 1, -g.count),
        )
        for group in sorted_groups:
            logging.info(f"  {group.issue_id}: {group.count} occurrences")

        logging.info(f"Bug directories created in: {self.output.output_dir}")
        if self.action:
            logging.info(f"Action '{self.action}' performed for all directories")


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    log_file = f"x-vsf-find-bugs.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%y%m%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )

    logging.info(f"Log file: {log_file}")


# ============================================================================
# CLI
# ============================================================================


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Parse VASP OUTCAR files for errors and warnings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/vasp/jobs
  %(prog)s /path/to/vasp/jobs --symlink
  %(prog)s /path/to/vasp/jobs --copy
  %(prog)s /path/to/vasp/jobs -p "conv_*"
        """,
    )

    parser.add_argument(
        "base_dir", help="Base directory containing VASP job subdirectories", type=Path
    )

    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern to filter subdirectories (default: '*')",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--copy",
        action="store_true",
        help="Copy directories to bug_groups/bug_name/dirs/",
    )
    group.add_argument(
        "--move",
        action="store_true",
        help="Move directories to bug_groups/bug_name/dirs/",
    )
    group.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links in bug_groups/bug_name/dirs/",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate base directory
    if not args.base_dir.exists():
        logging.info(f"Error: Directory {args.base_dir} does not exist")
        return 1

    if not args.base_dir.is_dir():
        logging.info(f"Error: {args.base_dir} is not a directory")
        return 1

    # Determine action
    action = None
    if args.copy:
        action = "copy"
    elif args.move:
        action = "move"
    elif args.symlink:
        action = "symlink"

    # Run pipeline
    try:
        pipeline = VaspAnalysisPipeline(
            args.base_dir, pattern=args.pattern, action=action
        )
        pipeline.run()
        return 0
    except Exception as e:
        logging.info(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
