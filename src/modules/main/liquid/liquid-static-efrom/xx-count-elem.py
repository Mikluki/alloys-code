from collections import Counter
from pathlib import Path


def count_element_files(directory):
    """
    Count files by element from decorr_ELEMENT_NUMBER pattern.

    Args:
        directory: Path to the directory containing the files
    """
    # Get all files in directory
    dir_path = Path(directory)
    files = [f.name for f in dir_path.iterdir() if f.is_dir()]

    # Extract elements from files starting with "decorr"
    elements = []
    for filename in files:
        if filename.startswith("decorr_"):
            parts = filename.split("_")
            if len(parts) >= 2:
                element = parts[1]  # Get element name between underscores
                elements.append(element)

    # Count occurrences
    element_counts = Counter(elements)

    # Print statistics
    print(f"Total files analyzed: {len(elements)}")
    print(f"Unique elements: {len(element_counts)}")
    print("\nCounts by element:")
    for element, count in sorted(element_counts.items()):
        print(f"  {element}: {count}")

    return element_counts


# Usage
if __name__ == "__main__":
    directory = "x-all300k-decor-poscar"
    counts = count_element_files(directory)
