## FEW NBANDS

relaunch with more nbands, look vasp wiki
add NBANDS to incar

```
/Re4_W4_ml-1000080_
 -----------------------------------------------------------------------------
|                                                                             |
|           W    W    AA    RRRRR   N    N  II  N    N   GGGG   !!!           |
|           W    W   A  A   R    R  NN   N  II  NN   N  G    G  !!!           |
|           W    W  A    A  R    R  N N  N  II  N N  N  G       !!!           |
|           W WW W  AAAAAA  RRRRR   N  N N  II  N  N N  G  GGG   !            |
|           WW  WW  A    A  R   R   N   NN  II  N   NN  G    G                |
|           W    W  A    A  R    R  N    N  II  N    N   GGGG   !!!           |
|                                                                             |
|     Your highest band is occupied at some k-points! Unless you are          |
|     performing a calculation for an insulator or semiconductor, without     |
|     unoccupied bands, you have included TOO FEW BANDS!! Please increase     |
|     the parameter NBANDS in file INCAR to ensure that the highest band      |
|     is unoccupied at all k-points. It is always recommended to include      |
|     a few unoccupied bands to accelerate the convergence of                 |
|     molecular-dynamics runs (even for insulators or semiconductors),        |
|     since the presence of unoccupied bands improves wavefunction            |
|     prediction and helps to suppress 'band-crossings'.                      |
|                                                                             |
 -----------------------------------------------------------------------------
```

##

```
B1--Zr4_Re4_ml-1000201_
 -----------------------------------------------------------------------------
|                                                                             |
|           W    W    AA    RRRRR   N    N  II  N    N   GGGG   !!!           |
|           W    W   A  A   R    R  NN   N  II  NN   N  G    G  !!!           |
|           W    W  A    A  R    R  N N  N  II  N N  N  G       !!!           |
|           W WW W  AAAAAA  RRRRR   N  N N  II  N  N N  G  GGG   !            |
|           WW  WW  A    A  R   R   N   NN  II  N   NN  G    G                |
|           W    W  A    A  R    R  N    N  II  N    N   GGGG   !!!           |
|                                                                             |
|     The electronic self-consistency was not achieved in the given           |
|     number of steps (NELM). The forces and other quantities evaluated       |
|     might not be reliable so examine the results carefully. If you find     |
|     spurious results, we suggest increasing NELM, if you were close to      |
|     convergence or switching to a different ALGO or adjusting the           |
|     density mixing parameters otherwise.                                    |
|                                                                             |
 -----------------------------------------------------------------------------

```

```
Re4_W4_ml-1000080_/OUTCAR
 -----------------------------------------------------------------------------
|                                                                             |
|     EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR      ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     EEEEE    RRRRRR   RRRRRR   O     O  RRRRRR       #       #       #      |
|     E        R   R    R   R    O     O  R   R                               |
|     E        R    R   R    R   O     O  R    R      ###     ###     ###     |
|     EEEEEEE  R     R  R     R  OOOOOOO  R     R     ###     ###     ###     |
|                                                                             |
|     ZBRENT: fatal error in bracketing                                       |
|      please rerun with smaller EDIFF, or copy CONTCAR                       |
|      to POSCAR and continue                                                 |
|                                                                             |
|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |
|                                                                             |
 -----------------------------------------------------------------------------
```
