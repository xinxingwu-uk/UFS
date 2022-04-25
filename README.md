# Codes for "Algorithmic Stability and Generalization of an Unsupervised Feature Selection Algorithm"

> Feature selection, as an important dimension reduction technique, reduces data dimension by identifying an essential subset of input features, which can facilitate interpretable insights into learning and inference processes. Algorithmic stability is a key characteristic of an algorithm regarding its sensitivity to perturbations of input samples. In this paper, we propose an innovative unsupervised feature selection algorithm attaining this stability with provable guarantees. The architecture of our algorithm consists of a feature scorer and a feature selector. The scorer trains a neural network (NN) to globally score all the features, and the selector adopts a dependent sub-NN to locally evaluate the representation abilities for selecting features. Further, we present algorithmic stability analysis and show that our algorithm has a performance guarantee via a generalization error bound. Extensive experimental results on real-world datasets demonstrate superior generalization performance of our proposed algorithm to strong baseline methods. Also, the properties revealed by our theoretical analysis and the stability of our algorithm-selected features are empirically confirmed.

---
All experiments are implemented by JupyterLab 2.2.4 with Python 3.7.8, Tensorflow 1.14, and Keras 2.2.5. The files in the subfolder “Python” are the Python source codes, which have been implemented in JupyterLab. For readability, we also provide the corresponding html files in the subfolder “Html”.

1.The codes in folder "Assumption2" are for the verification of Assumption 2.

2.The codes in folder "Square" are for the results in Tables 2 and 3 when $\Phi(\mathrm{W}_{\mathrm{I}})^{\mathrm{max}_k}=\mathrm{W}_{\mathrm{I}}^2$.

3.The codes in folder "AbsoluteAndOtherAlgorithms" are for the results in Tables 2 and 3 when $$\Phi(\mathrm{W}_{\mathrm{I}})^{\mathrm{max}_k}=|\mathrm{W}_{\mathrm{I}}|.$$

4.The codes in folder "InterpretationOfBound" are for Figure 2.

5.The codes in folder "StabilityAnalysis" are for Figure 2 in the main text, and they are the experiments related to stability analysis.

---

**If you find this code useful in your research, please consider citing our work:**

Xinxing Wu and Qiang Cheng. Algorithmic stability and generalization of an unsupervised feature selection algorithm. Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021); preprint arXiv: 2010.09416v1, 2021.


---

## License

Distributed under the MIT license. See [``LICENSE``](https://github.com/xinxingwu-uk/FAE/blob/main/LICENSE) for more information.
