## Structure

### [MACE](https://github.com/ACEsuit/mace?tab=readme-ov-file#installation-from-pypi)

> ez to install

```bash
uv venv ~/.venvs/uv312mace --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

uv pip install mace-torch
```

### eSEN

> meta did breaking changes migrating to version 2.0 In the future deprications are likely

```bash
uv venv ~/.venvs/uv312eSEN --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install fairchem-core
```

### Nequip

> [!NOTE] Destribution is such a pain in the ass. I do not think it can be built without cuda device

```bash
uv venv ~/.venvs/uv312Nequip --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

nequip-compile \
  nequip.net:mir-group/NequIP-OAM-L:0.1 \
  mir-group__NequIP-OAM-L__0.1.nequip.pt2 \
  --mode aotinductor \
  --device cpu \
  --target ase
```

### ORB v3

> ez to install

```bash
uv venv ~/.venvs/uv312ORBv3 --python 3.12
```

### SevenNet

```bash
uv venv ~/.venvs/uv312SevenNet --python 3.12

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

```

## For each
```bash
uv pip install -e .
```

## Alias

```bash
alias uv312mace="source /home/mik/.venvs/uv312mace/bin/activate"
alias uv312eSEN="source /home/mik/.venvs/uv312eSEN/bin/activate"
alias uv312Nequip="source /home/mik/.venvs/uv312Nequip/bin/activate"
alias uv312ORBv3="source /home/mik/.venvs/uv312ORBv3/bin/activate"
alias uv312SevenNet="source /home/mik/.venvs/uv312SevenNet/bin/activate"
```
