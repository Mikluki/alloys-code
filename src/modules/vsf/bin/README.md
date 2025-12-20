# VASP input parameters

## MP double relaxation

Based on Materials Project convention, here are the parameters for **both relaxations**:

### First Relaxation:

```python
{
    "IBRION": 2,
    "NSW": 99,
    "ISIF": 3,
    "EDIFFG": -0.02,
    "LCHARG": False,  # Don't write CHGCAR
    "LWAVE": True     # Write WAVECAR for second run
}
```

### Second Relaxation:

```python
{
    "IBRION": 2,
    "NSW": 99,
    "ISIF": 3,
    "EDIFFG": -0.02,
    "ISTART": 1,
    "ICHARG": 0,
    "LCHARG": True,
}
```

# VSF Command Line Tools

VSF includes command line utilities for common file operations after parallel structure generation.

## Setup

### One-time (current session only)

```bash
export VSF_BIN=$(python3 -c "import vsf; print(vsf.__path__[0])")/bin
export PATH="$VSF_BIN:$PATH"
```

### Permanent (all future sessions)

```bash
# Get VSF bin directory
export VSF_BIN=$(py_env -c "import vsf; print(vsf.__path__[0])")/bin

# Add to your shell config (choose one):
echo 'export PATH="'$VSF_BIN':$PATH"' >> ~/.bashrc   # For bash
echo 'export PATH="'$VSF_BIN':$PATH"' >> ~/.zshrc    # For zsh

# Load in current session
source ~/.bashrc   # or source ~/.zshrc
```

## Usage

To see available commands:

```bash
ls $VSF_BIN
# or use tab completion
vsf<TAB><TAB>
```
