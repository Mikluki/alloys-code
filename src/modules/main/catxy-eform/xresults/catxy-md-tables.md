<!-- ```bash -->
<!-- [INFO] core.analysis: Loaded 5220 structure results from 4 directories -->
<!-- [INFO] core.analysis: Selected 1445 structures below threshold 0.01 eV/atom -->
<!-- [INFO] core.analysis: Selected 1582 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: MACE: 1363/1445 correctly classified structures -->
<!-- [INFO] vsf: MACE: 82 stable structures misclassified as unstable -->
<!-- [INFO] vsf: MACE: 219 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 1534 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: MACE_MPA_0: 1365/1445 correctly classified structures -->
<!-- [INFO] vsf: MACE_MPA_0: 80 stable structures misclassified as unstable -->
<!-- [INFO] vsf: MACE_MPA_0: 169 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 1681 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: ESEN_30M_OAM: 1379/1445 correctly classified structures -->
<!-- [INFO] vsf: ESEN_30M_OAM: 66 stable structures misclassified as unstable -->
<!-- [INFO] vsf: ESEN_30M_OAM: 302 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 1513 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: ORBV3: 1366/1445 correctly classified structures -->
<!-- [INFO] vsf: ORBV3: 79 stable structures misclassified as unstable -->
<!-- [INFO] vsf: ORBV3: 147 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 1506 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: SEVENNET: 1373/1445 correctly classified structures -->
<!-- [INFO] vsf: SEVENNET: 72 stable structures misclassified as unstable -->
<!-- [INFO] vsf: SEVENNET: 133 unstable structures misclassified as stable -->
<!-- ``` -->


\begin{table}[htbp]
\centering
\begin{tabular}{lccc}
\toprule
\multicolumn{4}{c}{GNN Model Performance vs VASP Reference} \\
GNN Model & \makecell{Correctly Classified\\Stable Structures} & \makecell{Stable Structures\\Misclassified as Unstable} & \makecell{Unstable Structures\\Misclassified as Stable} \\
\midrule
ESEN     & 1379/1445 & 66  & 302 \\
SevenNet & 1373/1445 & 72  & 133 \\
ORB      & 1366/1445 & 79  & 147 \\
MACE-MPA & 1365/1445 & 80  & 169 \\
MACE     & 1363/1445 & 82  & 219 \\
\bottomrule
\end{tabular}
\end{table}


<!-- | GNN Model | MAE (eV/atom) | R² | Performance Rank | -->
<!-- |--------------|-------|------|-----| -->
<!-- | SevenNet     | 0.090 | 0.96 | 1st | -->
<!-- | MACE-MPA   | 0.093 | 0.96 | 2nd | -->
<!-- | MACE         | 0.106 | 0.96 | 3rd | -->
<!-- | ORB        | 0.116 | 0.95 | 4th | -->
<!-- | ESEN | 0.337 | 0.41 | 5th | -->


\begin{table}[htbp]
\centering
\begin{tabular}{lccc}
\toprule
GNN Model & MAE (eV/atom) & R$^2$ & Performance Rank \\
\midrule
SevenNet   & 0.090 & 0.96 & 1st \\
MACE-MPA   & 0.093 & 0.96 & 2nd \\
MACE       & 0.106 & 0.96 & 3rd \\
ORB        & 0.116 & 0.95 & 4th \\
ESEN       & 0.337 & 0.41 & 5th \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\begin{tabular}{lcccccc}
\toprule
GNN Model & \makecell{MAE\\(eV/atom)} & \makecell{Correctly Classified\\Stable Structures} & \makecell{Stable Structures\\Misclassified as Unstable} & \makecell{Unstable Structures\\Misclassified as Stable} \\
\midrule
SevenNet   & 0.090 & 1373/1445 & 72 & 133 \\
MACE-MPA   & 0.093 & 1365/1445 & 80 & 169 \\
MACE       & 0.106 & 1363/1445 & 82 & 219 \\
ORB        & 0.116 & 1366/1445 & 79 & 147 \\
ESEN       & 0.337 & 1379/1445 & 66 & 302 \\
\bottomrule
\end{tabular}
\end{table}


\newpage

| Model        | Matbench Discovery        | Our Dataset              |
| ------------ | ------------------------- | ------------------------ |
| **ESEN**     | R² = 0.87, MAE = 0.018 eV | R² = 0.41, MAE = 0.34 eV |
| **SevenNet** | R² = 0.87, MAE = 0.021 eV | R² = 0.96, MAE = 0.09 eV |

| Dataset        | Structures | Composition                   | Experimental Relevance        |
| -------------- | ---------- | ----------------------------- | ----------------------------- |
| Binary Alloys  | 5,220      | TM-TM pairs from prototypes   | High - feasible for lab synthesis |
| Random 16-atom | 5,000      | Random TM arrangements        | Medium - composition space exploration |
| Liquid 64-atom | ~3,000     | Al, Na, Au$_4$, Cu$_4$, N$_4$ | Low - GNN stress test, unconventional structures |



<!-- \begin{table}[htbp] -->
<!-- \centering -->
<!-- \begin{tabular}{lcc} -->
<!-- \toprule -->
<!-- GNN Model & \makecell{} -->
<!-- \midrule -->
<!-- \bottomrule -->
<!-- \end{tabular} -->
<!-- \end{table} -->
