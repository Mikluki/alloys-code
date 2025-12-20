"""
Structure organization with unique ID assignment.
Wraps PoscarOrganizer to add energy.json support.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List

from vsf.liquid.extract import ConfigurationData
from vsf.liquid.savers import save_poscar_only, save_structure_with_energy
from vsf.transform.poscar_organizer import PoscarOrganizer

LOGGER = logging.getLogger(__name__)


class StructureOrganizer:
    """
    Wrapper around PoscarOrganizer that handles energy.json integration.
    Separates ID assignment from file saving.
    """

    def __init__(self, starting_id: int, prefix: str = "liquid"):
        """
        Initialize organizer.

        Args:
            starting_id: Starting ID for structure numbering
            prefix: Prefix for structure IDs (e.g., "liquid" -> "liquid_2002001")
        """

        self.organizer = PoscarOrganizer.from_starting_id(starting_id)
        self.prefix = prefix
        self.id_mapping: Dict[str, ConfigurationData] = {}

    def organize_configs(
        self, configs: List[ConfigurationData], target_dir: Path
    ) -> Dict[str, Path]:
        """
        Organize configurations with unique IDs using PoscarOrganizer.

        Process:
        1. Create temporary POSCARs
        2. Let PoscarOrganizer assign IDs and create directories
        3. Add energy.json to each directory
        4. Clean up temporary files

        Args:
            configs: List of configurations to organize
            target_dir: Target directory for organized structures

        Returns:
            Dictionary mapping structure_id -> directory_path
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Organizing {len(configs)} configurations with IDs...")
        LOGGER.info(f"Target directory: {target_dir}")
        LOGGER.info(f"Starting ID: {self.organizer.starting_id}, Prefix: {self.prefix}")

        # Step 1: Create temporary POSCARs (required by PoscarOrganizer)
        temp_dir = Path("_temp_org")
        temp_dir.mkdir(exist_ok=True)

        temp_poscars = []
        for config in configs:
            temp_path = temp_dir / f"temp_{config.config_id}.poscar"
            save_poscar_only(config, temp_path)
            temp_poscars.append(temp_path)

        LOGGER.debug(f"Created {len(temp_poscars)} temporary POSCARs")

        # Step 2: Use PoscarOrganizer to assign IDs and create structure
        organized = self.organizer.organize_poscar_list(
            temp_poscars, target_dir, self.prefix
        )

        LOGGER.info(f"PoscarOrganizer created {len(organized)} directories")

        # Step 3: Add energy.json to each organized directory
        for i, (struct_id, org_dir) in enumerate(organized.items()):
            config = configs[i]
            self.id_mapping[struct_id] = config

            # Add energy.json with structure_id in metadata
            save_structure_with_energy(
                config, org_dir, metadata_extra={"structure_id": struct_id}
            )

        # Step 4: Cleanup temporary files
        shutil.rmtree(temp_dir)
        LOGGER.debug("Cleaned up temporary directory")

        LOGGER.info(f"Successfully organized {len(organized)} structures with IDs")
        return organized
