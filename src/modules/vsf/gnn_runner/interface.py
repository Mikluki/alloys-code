"""
GNN Runner CLI Interface and Configuration.

Architecture
------------
The GNN runner is invoked both via CLI and programmatically (from GNNEnergyWorkflow).
To maintain consistency:

1. CLI args are defined in `create_parser()`
2. Config dataclass mirrors these exactly in `GNNRunnerConfig`
3. `validate_interface_compatibility()` ensures they stay in sync (runs at import)

Adding New Arguments
--------------------
When adding a new argument, you MUST update THREE places:

1. **Parser** (`create_parser`):
   Add the argument definition:
```python
   parser.add_argument(
       "--my-new-arg",
       default="value",
       type=str,
       help="Description of my new argument"
   )
```

2. **Dataclass** (`GNNRunnerConfig`):
   Add the field with matching name and type:
```python
   @dataclass
   class GNNRunnerConfig:
       ...
       my_new_arg: str = "value"
```

3. **Conversion && CLI Serialization**
   `from_args` method:
   Add to the constructor call
   ```python
      return cls(
          ...
          my_new_arg=args.my_new_arg,
      )
   ```

   `to_cli_args` method:
   Add logic to convert back to CLI args
   ```python
      args.extend(["--my-new-arg", self.my_new_arg])
   ```

The `validate_interface_compatibility()` function will catch mismatches between
the parser and dataclass at import time, preventing drift.

Example Usage
-------------
Programmatic (from GNNEnergyWorkflow):
    config = GNNRunnerConfig(
        energy_source="MACE",
        structure_list=Path("structures.txt"),
        device="cuda:0"
    )
    cmd = ["python", "-m", "vsf.gnn_runner"] + config.to_cli_args()
    subprocess.run(cmd)

Command Line:
    python -m vsf.gnn_runner --energy-source MACE --structure-list structures.txt
"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, List


@dataclass
class GNNRunnerConfig:
    """Interface contract for vsf.gnn_runner subprocess."""

    energy_source: str
    structure_list: Path
    device: str = "cpu"
    overwrite: bool = False
    cleanup_deprecated: bool = False
    progress_interval: int = 10
    checkpoint_path: Path | None = None
    save_energy: bool = True
    save_stress: bool = True

    def to_cli_args(self) -> List[str]:
        """Convert to subprocess CLI arguments."""
        args = [
            "--energy-source",
            self.energy_source,
            "--structure-list",
            str(self.structure_list),
            "--device",
            self.device,
            "--progress-interval",
            str(self.progress_interval),
        ]

        if self.overwrite:
            args.append("--overwrite")
        if self.cleanup_deprecated:
            args.append("--cleanup-deprecated")
        if self.checkpoint_path:
            args.extend(["--checkpoint-path", str(self.checkpoint_path)])
        # Set save Flags
        if self.save_energy:
            args.append("--save-energy")
        if self.save_stress:
            args.append("--save-stress")

        return args

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GNNRunnerConfig":
        """Create from parsed CLI arguments."""
        return cls(
            energy_source=args.energy_source,
            structure_list=args.structure_list,
            device=args.device,
            overwrite=args.overwrite,
            cleanup_deprecated=args.cleanup_deprecated,
            progress_interval=args.progress_interval,
            checkpoint_path=args.checkpoint_path,
            save_energy=args.save_energy,
            save_stress=args.save_stress,
        )

    def to_calc_kwargs(self) -> dict[str, Any]:
        """Extract calculator initialization kwargs."""
        kwargs = {"device": self.device}
        if self.checkpoint_path:
            kwargs["checkpoint_path"] = str(self.checkpoint_path)
        return kwargs


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for GNN runner."""
    parser = argparse.ArgumentParser(
        description="GNN batch energy calculator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--energy-source",
        required=True,
        type=str,
        help="Energy source/calculator type (e.g., MACE, CHGNet)",
    )
    parser.add_argument(
        "--structure-list",
        required=True,
        type=Path,
        help="File containing list of structure directory paths",
    )

    # Optional arguments
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing energy sources in JSON",
    )
    parser.add_argument(
        "--cleanup-deprecated",
        action="store_true",
        help="Whether to remove deprecated energy sources from JSON",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for calculation (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--progress-interval",
        default=10,
        type=int,
        help="Log progress every N structures",
    )

    # Calculator-specific kwargs
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Path to model checkpoint",
    )

    # Calculation request flags
    parser.add_argument(
        "--save-energy",
        action="store_true",
        default=True,
        help="Whether to calculate and save energy",
    )
    parser.add_argument(
        "--save-stress",
        action="store_true",
        default=True,
        help="Whether to calculate and save stress",
    )

    return parser


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args()


def validate_interface_compatibility() -> None:
    """Ensure CLI parser and dataclass are in sync."""
    parser = create_parser()
    parser_args = {action.dest for action in parser._actions if action.dest != "help"}
    dataclass_fields = {field.name for field in fields(GNNRunnerConfig)}

    missing_in_cli = dataclass_fields - parser_args
    missing_in_dataclass = parser_args - dataclass_fields

    if missing_in_cli or missing_in_dataclass:
        raise RuntimeError(
            f"Interface mismatch! "
            f"CLI missing: {missing_in_cli}, "
            f"Dataclass missing: {missing_in_dataclass}"
        )
