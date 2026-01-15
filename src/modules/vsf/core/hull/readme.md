### **Equation of state**

The EOS benchmark protocol in Arena includes first unconstrained structure optimization at $0 , \mathrm{K}$ and subsequently multiple energy calculations of isotropic deformations, including ionic relaxation at volumetric strain ranging from $-20%$ to $20%$ of the optimized structure. After ionic relaxation of $21$ deformed structures for each crystal, Birch–Murnaghan EOS is fitted with the following equation:

$$
E = E_0 + \frac{9 B V_0}{16}
\left[
\left(\eta^2 - 1\right)^2
\left(
6 + B' \left(\eta^2 - 1\right) - 4\eta^2
\right)
\right],
\qquad
\eta = \left( \frac{V}{V_0} \right)^{1/3}
\tag{S4}
$$

where $V_0$ is the equilibrium volume after initial structure optimization, and $B$ and $B'$ are the bulk modulus and its pressure derivative from the EOS fit by rearranging Equation $(\mathrm{S4})$ as:

$$
\frac{\Delta E}{B V_0} =
\frac{E - E_0}{B V_0} =
\frac{9}{16}
\left[
\left(\eta^2 - 1\right)^2
\left(
6 + B' \left(\eta^2 - 1\right) - 4 \eta^2
\right)
\right]
$$
