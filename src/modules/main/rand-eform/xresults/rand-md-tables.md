<!-- ```bash -->
<!-- [INFO] vsf: MACE: 23/37 correctly classified structures -->
<!-- [INFO] vsf: MACE: 14 stable structures misclassified as unstable -->
<!-- [INFO] vsf: MACE: 2462 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 2618 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: MACE_MPA_0: 25/37 correctly classified structures -->
<!-- [INFO] vsf: MACE_MPA_0: 12 stable structures misclassified as unstable -->
<!-- [INFO] vsf: MACE_MPA_0: 2593 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 2500 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: ESEN_30M_OAM: 23/37 correctly classified structures -->
<!-- [INFO] vsf: ESEN_30M_OAM: 14 stable structures misclassified as unstable -->
<!-- [INFO] vsf: ESEN_30M_OAM: 2477 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 2487 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: ORBV3: 23/37 correctly classified structures -->
<!-- [INFO] vsf: ORBV3: 14 stable structures misclassified as unstable -->
<!-- [INFO] vsf: ORBV3: 2464 unstable structures misclassified as stable -->
<!-- [INFO] core.analysis: Selected 2515 structures below threshold 0.01 eV/atom -->
<!-- [INFO] vsf: SEVENNET: 23/37 correctly classified structures -->
<!-- [INFO] vsf: SEVENNET: 14 stable structures misclassified as unstable -->
<!-- [INFO] vsf: SEVENNET: 2492 unstable structures misclassified as stable -->
<!-- ``` -->


\begin{table}[htbp]
\centering
\begin{tabular}{lccc}
\toprule
\multicolumn{4}{c}{GNN Model Performance vs VASP Reference} \\
\midrule
GNN Model & \makecell{Correctly Classified\\Stable Structures} & \makecell{Stable Structures\\Misclassified as Unstable} & \makecell{Unstable Structures\\Misclassified as Stable} \\
\midrule
MACE-MPA & 25/37 & 12  & 2593 \\
MACE     & 23/37 & 14  & 2462 \\
ESEN     & 23/37 & 14  & 2477 \\
ORB      & 23/37 & 14  & 2464 \\
SevenNet & 23/37 & 14  & 2492 \\
\bottomrule
\end{tabular}
\end{table}


\begin{table}[htbp]
\centering
\begin{tabular}{lccc}
\toprule
\multicolumn{4}{c}{GNN Model Performance vs VASP Reference} \\
\midrule
GNN Model & \makecell{Correctly Classified\\Stable Structures} & \makecell{Unstable Structures\\Misclassified as Stable} & \makecell{MAE\\(eV/atom)} \\
\midrule
MACE-MPA & 25/37 & 2593 & 0.625 \\
MACE     & 23/37 & 2462 & 0.603 \\
ESEN     & 23/37 & 2477 & 0.604 \\
ORB      & 23/37 & 2464 & 0.605 \\
SevenNet & 23/37 & 2492 & 0.606 \\
\bottomrule
\end{tabular}
\end{table}


| GNN Model    | MAE (eV/atom) |
| ------------ | ------------- |
| MACE         | 0.603         |
| ESEN_30M_OAM | 0.604         |
| ORBV3        | 0.605         |
| SEVENNET     | 0.606         |
| MACE_MPA_0   | 0.625         |
