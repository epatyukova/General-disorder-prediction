# General-disorder-prediction
This repository contains models used to predict general disorder of compounds (ordered/disordered, binary classification)

At the most basic level all compounds can be divided into two categories: ordered and disordered, judging by the presence of the partial occupancies in CIF. We start from trying to find the best machine learning model which would predict disorder from the composition. 

Table~\ref{models-global-disorder} describes some of the best models we were able to build together with their performance. 

\begin{table}[H]
\centering
\begin{tabular}{||l|l|l|c|c|c|c|c||}
\hline 
\# & Model & Features & Balanced Accuracy & Recall & Precision & ROC AUC & MCC  \\ \hline
1& RF & Magpie & 0.87 & 0.90 & 0.90 & 0.94 & 0.74 \\ \hline
2& 3-NN & Magpie (scaled) & 0.79 & 0.83 & 0.83 & 0.86 & 0.58 \\ \hline
3& Roost & Matscholar & 0.84 & 0.83 & 0.89 & 0.91 & 0.67 \\ \hline
4& CrabNet & Mat2vec & 0.90 & 0.88 & 0.94 & 0.95 & 0.79 \\ \hline
\end{tabular}
\caption{Models for general disorder prediction. Models were trained using the same training-validation-test split. Abbreviations used are: RF = Random Forrest, 3-NN = Nearest Neighbor Classifier with 3 neighbors, ROC AUC = area under the receiver operator curve, MCC = Matthews correlation coefficient.}
\label{models-global-disorder}
\end{table}
