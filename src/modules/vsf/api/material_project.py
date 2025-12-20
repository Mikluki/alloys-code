import logging
from typing import List, Optional

from emmet.core.thermo import ThermoType
from mp_api.client import MPRester

from ..energy.energy_source import EnergySource
from ..entries.entry import StructureEntry
from ..entries.entry_manager import EntryManager

LOGGER = logging.getLogger(__name__)

DEFAULT_THERMO_TYPES = [ThermoType.GGA_GGA_U]


class MaterialsProjectFetcher:
    """
    A class to handle fetching `structures` & `energy_data` from the Materials Project API.

    Attributes:
        api_key (str): API key for accessing the Materials Project database.
        thermo_types (list): Types of thermodynamic calculations to fetch.
    """

    def __init__(
        self,
        api_key: str,
        thermo_types: List[ThermoType] | None = DEFAULT_THERMO_TYPES,
    ):
        """
        Initialize the fetcher with Materials Project API key.

        Args:
            api_key (str): Materials Project API key
            thermo_types (Optional[List[ThermoType]]): List of thermodynamic calculation types.
                Defaults to ThermoType.GGA_GGA_U enums.
                If "all" [ThermoType.GGA_GGA_U, ThermoType.GGA_GGA_U_R2SCAN, ThermoType.R2SCAN]
        """
        self.api_key = api_key
        self.thermo_types = thermo_types
        self.source = EnergySource.MP

        if thermo_types == "all":
            thermo_types = [
                ThermoType.GGA_GGA_U,
                ThermoType.GGA_GGA_U_R2SCAN,
                ThermoType.R2SCAN,
            ]

    def _fetch_summary_data(self, material_ids: list) -> Optional[list]:
        """
        Fetch thermodynamic data from Materials Project API.

        Args:
            material_ids: List of Material Project IDs

        Returns:
            List of thermodynamic data documents or None if fetch fails
        """
        try:
            with MPRester(api_key=self.api_key) as mpr:
                return mpr.materials.thermo.search(
                    thermo_types=self.thermo_types,  # pyright: ignore
                    material_ids=material_ids,
                )
        except Exception as e:
            LOGGER.error(f"Failed to fetch data from MP API: {e}")
            return None

    def _update_entry(
        self,
        entry,
        mp_id: str,
        potential_energy_pa: float,
        formation_energy_pa: float,
        save_json: bool = True,
    ) -> None:
        """
        Update entry with fetched energy data.

        Args:
            entry: Entry object to update
            mp_id: Materials Project ID
            potential_energy_pa: Potential energy per atom
            formation_energy_pa: Formation energy per atom
            save_json: Whether to save the entry as JSON
        """
        entry.potential_energy.add_result(self.source, potential_energy_pa)
        entry.formation_energy.add_result(self.source, formation_energy_pa)

        if entry.pure_structure:
            LOGGER.info(
                f"For {mp_id} [{entry.name}] Fetched E_potential = {potential_energy_pa}"
            )
        else:
            LOGGER.info(
                f"For {mp_id} [{entry.name}] Fetched E_potential = {potential_energy_pa}"
            )
            LOGGER.info(
                f"For {mp_id} [{entry.name}] Fetched E_formation = {formation_energy_pa}"
            )

        if save_json:
            entry.save_json()

    def fetch_energy(
        self, entries: List["StructureEntry"], save_json: bool = True, refetch=False
    ) -> None:
        """
        Fetch energy per atom and formation energy per atom for a list of entries.
        Updates the entries in place with the fetched data.

        Warning: Do not call consecutively! Can cause packet loss presumably due to rate limit

        Args:
            entries: List of StructureEntry objects to update with energy data
            save_json: Whether to save updated entries as JSON
        """
        if not entries:
            LOGGER.info("No Fetch via MP_API. entries list is empty.")
            return

        # Check entries to have MP_ID type
        EntryManager.validate_mp_entries(entries)

        if refetch is True:
            entries_without_energy = entries
        else:
            entries_without_energy = EntryManager.find_unfetched_entries(
                entries, self.source
            )

        if not entries_without_energy:
            LOGGER.info("!! Skip Fetch via MP_API. All entries have energies.")
            return

        mp_ids = [entry.id for entry in entries_without_energy]
        LOGGER.info(f"Fetching energy data for {mp_ids} via MP_API.")

        thermo_docs = self._fetch_summary_data(mp_ids)
        if not thermo_docs:
            LOGGER.error(f"Thermo_docs {thermo_docs} are empty.")
            return

        # Create a lookup for faster access
        entry_by_mp_id = {entry.id: entry for entry in entries_without_energy}

        for doc in thermo_docs:
            mp_id = doc.material_id  # pyright: ignore
            potential_energy_pa = doc.energy_per_atom  # pyright: ignore
            formation_energy_pa = doc.formation_energy_per_atom  # pyright: ignore

            entry = entry_by_mp_id.get(mp_id)
            if entry:
                self._update_entry(
                    entry, mp_id, potential_energy_pa, formation_energy_pa, save_json
                )
            else:
                LOGGER.error(f"Fetched {mp_id}, but it has no Entry")
