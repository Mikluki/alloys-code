## Note

the difference between makefiles is only one line:

```bash
VASP_TARGET_CPU ?= -xICELAKE-SERVER
VASP_TARGET_CPU ?= -xCORE-AVX512
VASP_TARGET_CPU ?= -xCORE-AVX2
```

The flags were gpted, but they appear to work as intended.

If you think that this one is a better idea:

```bash
VASP_TARGET_CPU ?= -xHOST
```

think twice, cause for this to work you have to send the build to corresponding node, otherwise you will end up optimizing for you guest node 🙃

### Docs

Running ifort `--help` or `icc -help` will list available -x… and -march=… options tailored to that version.

The [docs](https://www.vasp.at/wiki/index.php/Makefile.include.oneapi) unfortunately do not say much on the specific flags. Also this line from my experiments slows down execution considerably:

```bash
FCL        += -qmkl=sequential
```
