# VASP OUTCAR Parser - Refactoring Design

## Current Issues

### Critical Bug
- **Only classifies first warning**: When a directory has multiple warnings, only the first one is captured and used for classification
- **Ignores warnings when errors exist**: Warnings are completely ignored if any error is present

### Architectural Problems
1. **God Class**: `VaspOutcarParser` does everything (parsing, I/O, analysis, reporting)
2. **Tight Coupling**: Business logic mixed with file system operations
3. **Poor Testability**: Hard to unit test individual components
4. **Unclear Flow**: Single `run()` method obscures the actual process
5. **Limited Extensibility**: Adding features requires modifying core logic

## Proposed Architecture

### Design Principles
- **Single Responsibility**: Each class has one clear purpose
- **Separation of Concerns**: Parse → Analyze → Output pipeline
- **Data-Driven**: Use data classes to represent domain entities
- **Extensibility**: Easy to add new parsers, analyzers, or outputs

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Main Orchestrator                     │
│  (coordinates the pipeline, manages configuration)      │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│   Discover   │        │    Parse     │
│   (find dirs)│───────▶│  (extract)   │
└──────────────┘        └──────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │    Analyze    │
                       │  (classify)   │
                       └───────┬───────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐  ┌──────────┐
        │  Report  │   │   File   │  │   File   │
        │ (summary)│   │  Writer  │  │  Mover   │
        └──────────┘   └──────────┘  └──────────┘
```

## Data Models

### Core Entities

```python
@dataclass
class Issue:
    """Represents a single error or warning occurrence"""
    type: Literal["error", "warning"]
    pattern: str  # The actual pattern text
    context: str  # Full context (pattern + surrounding lines)
    line_number: int  # Where it occurred in the file

@dataclass
class VaspJob:
    """Represents a single VASP calculation directory"""
    path: Path
    name: str  # Directory name
    errors: List[Issue]  # ALL errors found
    warnings: List[Issue]  # ALL warnings found
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    @property
    def is_clean(self) -> bool:
        return not (self.has_errors or self.has_warnings)

@dataclass
class IssueGroup:
    """A group of jobs with the same issue"""
    issue_id: str  # e.g., "er1", "warn3"
    issue_type: Literal["error", "warning"]
    pattern: str  # The normalized/deduplicated pattern
    context: str  # Representative context
    jobs: List[VaspJob]
    
    @property
    def count(self) -> int:
        return len(self.jobs)
```

## Component Specifications

### 1. OutcarParser
**Responsibility**: Extract errors and warnings from OUTCAR files

```python
class OutcarParser:
    """Parses OUTCAR files to extract issues"""
    
    ERROR_PATTERN = "EEEEEEE  RRRRRR   RRRRRR"
    WARNING_PATTERN = "RRRRR   N    N  II  N    N   GGGG"
    CONTEXT_LINES = 11
    
    def parse_file(self, outcar_path: Path) -> Tuple[List[Issue], List[Issue]]:
        """
        Parse a single OUTCAR file.
        
        Returns:
            (all_errors, all_warnings) - Lists of Issue objects
        """
        
    def _extract_issues(self, content: str, pattern: str, 
                       issue_type: Literal["error", "warning"]) -> List[Issue]:
        """Extract all occurrences of a pattern with context"""
```

**Key Change**: Returns ALL errors and ALL warnings, not just the first one.

### 2. JobDiscovery
**Responsibility**: Find VASP job directories

```python
class JobDiscovery:
    """Discovers VASP job directories"""
    
    def __init__(self, base_dir: Path, pattern: str = "*_*"):
        self.base_dir = base_dir
        self.pattern = pattern
    
    def find_jobs(self) -> List[Path]:
        """Find all directories matching the pattern with OUTCAR files"""
        
    def validate_job(self, path: Path) -> bool:
        """Check if a directory is a valid VASP job"""
```

### 3. JobAnalyzer
**Responsibility**: Parse jobs and create VaspJob objects

```python
class JobAnalyzer:
    """Analyzes VASP jobs to extract issues"""
    
    def __init__(self, parser: OutcarParser):
        self.parser = parser
    
    def analyze_job(self, job_path: Path) -> VaspJob:
        """Analyze a single job directory"""
        
    def analyze_batch(self, job_paths: List[Path], 
                     progress_callback=None) -> List[VaspJob]:
        """Analyze multiple jobs with optional progress reporting"""
```

### 4. IssueClassifier
**Responsibility**: Group jobs by issues and assign IDs

```python
class IssueClassifier:
    """Classifies and groups issues across jobs"""
    
    def classify(self, jobs: List[VaspJob]) -> List[IssueGroup]:
        """
        Group jobs by unique issue patterns.
        
        Strategy:
        - Each unique error gets its own group (er1, er2, ...)
        - Each unique warning gets its own group (warn1, warn2, ...)
        - Jobs with multiple issues appear in multiple groups
        - Groups are sorted by frequency (most common first)
        """
        
    def _deduplicate_patterns(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group identical issue patterns together"""
        
    def _assign_ids(self, groups: List[IssueGroup]) -> None:
        """Assign IDs based on frequency (most common = lowest number)"""
```

**Key Change**: Jobs with multiple issues appear in multiple groups.

### 5. OutputManager
**Responsibility**: Generate all outputs (summary, directories, files)

```python
class OutputManager:
    """Manages all output generation"""
    
    def __init__(self, base_dir: Path, output_subdir: str = "0_py_bug_groups"):
        self.base_dir = base_dir
        self.output_dir = base_dir / output_subdir
    
    def write_summary(self, jobs: List[VaspJob], 
                     groups: List[IssueGroup]) -> Path:
        """Write comprehensive summary file"""
        
    def write_bad_vasprun(self, job: VaspJob) -> None:
        """Write BAD_VASPRUN file in the job directory"""
        
    def create_group_structure(self, groups: List[IssueGroup],
                              action: Optional[str] = None) -> None:
        """
        Create bug group directory structure.
        
        For each group:
        - Create bug_groups/{group_id}/
        - Write error_text.txt
        - Write directories.txt
        - Optionally copy/move/symlink directories to dirs/
        """
```

### 6. FileSystemOperations
**Responsibility**: Handle file operations (copy/move/symlink)

```python
class FileSystemOperations:
    """Handles file system operations"""
    
    @staticmethod
    def copy_directory(source: Path, target: Path) -> None:
        """Copy directory with error handling"""
        
    @staticmethod
    def move_directory(source: Path, target: Path) -> None:
        """Move directory with error handling"""
        
    @staticmethod
    def create_symlink(source: Path, target: Path) -> None:
        """Create symbolic link with error handling"""
        
    def perform_action(self, action: str, source: Path, target: Path) -> None:
        """Dispatch to appropriate method based on action"""
```

### 7. VaspAnalysisPipeline (Main Orchestrator)
**Responsibility**: Coordinate the entire analysis workflow

```python
class VaspAnalysisPipeline:
    """Main orchestrator for VASP analysis"""
    
    def __init__(self, base_dir: Path, action: Optional[str] = None):
        self.base_dir = base_dir
        self.action = action
        
        # Initialize components
        self.discovery = JobDiscovery(base_dir)
        self.parser = OutcarParser()
        self.analyzer = JobAnalyzer(self.parser)
        self.classifier = IssueClassifier()
        self.output = OutputManager(base_dir)
        self.fs_ops = FileSystemOperations()
    
    def run(self) -> AnalysisResults:
        """Execute the complete analysis pipeline"""
        # 1. Discover
        job_paths = self.discovery.find_jobs()
        
        # 2. Analyze
        jobs = self.analyzer.analyze_batch(job_paths, self._progress_callback)
        
        # 3. Write BAD_VASPRUN files
        for job in jobs:
            if not job.is_clean:
                self.output.write_bad_vasprun(job)
        
        # 4. Classify
        groups = self.classifier.classify(jobs)
        
        # 5. Output
        self.output.write_summary(jobs, groups)
        self.output.create_group_structure(groups, self.action)
        
        return AnalysisResults(jobs, groups)
```

## Classification Strategy Changes

### Current (Incorrect) Behavior
```
Job with: [error1, warning1, warning2]
  → Classified to: er1
  → warning1 and warning2 ignored

Job with: [warning1, warning2, warning3]
  → Classified to: warn1 (based on warning1 only)
  → warning2 and warning3 ignored
```

### New (Correct) Behavior
```
Job with: [error1, warning1, warning2]
  → Appears in: er1, warn1, warn2
  
Job with: [warning1, warning2, warning3]
  → Appears in: warn1, warn2, warn3
```

**Implications**:
- A single job can appear in multiple `directories.txt` files
- `summary.txt` shows all issues, not just primary ones
- More accurate understanding of issue co-occurrence

## File Structure

Single file organization:
```python
#!/usr/bin/env python3
"""
VASP OUTCAR Analysis Tool
Parses VASP OUTCAR files to identify and classify errors and warnings.
"""

# Imports
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal, Optional
from pathlib import Path
import subprocess
import shutil
import datetime
from collections import defaultdict

# Data Models
@dataclass
class Issue: ...
@dataclass  
class VaspJob: ...
@dataclass
class IssueGroup: ...
@dataclass
class AnalysisResults: ...

# Core Components
class OutcarParser: ...
class JobDiscovery: ...
class JobAnalyzer: ...
class IssueClassifier: ...
class OutputManager: ...
class FileSystemOperations: ...
class VaspAnalysisPipeline: ...

# CLI
def main(): ...

if __name__ == "__main__":
    main()
```

## Testing Strategy (Future)

Each component can be tested independently:
- **OutcarParser**: Test with sample OUTCAR content
- **JobDiscovery**: Test with mock directory structures
- **IssueClassifier**: Test with mock VaspJob objects
- **OutputManager**: Test output generation
- **Pipeline**: Integration tests

## Migration Path

1. ✅ Create design document (this file)
2. Implement data models
3. Implement OutcarParser with full extraction
4. Implement other components
5. Implement Pipeline orchestrator
6. Update CLI to use new Pipeline
7. Test with real data
8. Remove old VaspOutcarParser class

## Benefits of New Design

1. **Correctness**: Captures ALL warnings and errors
2. **Clarity**: Each component has a clear purpose
3. **Testability**: Can test each component in isolation
4. **Extensibility**: Easy to add new features:
   - Different output formats (JSON, CSV)
   - Different classification strategies
   - Additional analysis (statistics, correlations)
   - Web interface or GUI
5. **Maintainability**: Changes are localized to specific components
