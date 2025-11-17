# Copula-Based Fusion of Clinical and Genomic Machine Learning Risk Scores for Breast Cancer Risk Stratification 🧬📊

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%2Fxgboost-orange)](https://scikit-learn.org/stable/)
[![Lifelines](https://img.shields.io/badge/Survival-Lifelines-FF6F61)](https://lifelines.readthedocs.io/en/latest/)
[![Genomics](https://img.shields.io/badge/Bio-Genomics-2E8B57)](https://en.wikipedia.org/wiki/Genomics)
[![Copulas](https://img.shields.io/badge/Stats-Copula%20Modeling-6E40AA)](https://en.wikipedia.org/wiki/Copula_(probability_theory))
[![METABRIC](https://img.shields.io/badge/Data-METABRIC-blue)](https://www.cbioportal.org/study/summary?id=brca_metabric)

This repository contains the complete, reproducible machine learning pipeline for our study on integrating multi-view biomedical data (clinical and genomic) using Copula theory.

We introduce a fusion framework that models the dependency structure between disparate risk scores using Gaussian, Clayton, and Gumbel copulas. We evaluate the nature of joint risk dependence in breast cancer patients.

All results, methodological details, and discussions are provided in our accompanying research paper. This repository focuses strictly on code and reproducibility.

📂 Dataset Information
The analysis relies on the  processed METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset accessed via Kaggle.
**Source:** [Breast Cancer Gene Expression Profiles (METABRIC)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric/data?fbclid=IwY2xjawOHoQRleHRuA2FlbQIxMABicmlkETFkNEVlcUMzS1FCRTU5eE9Ic3J0YwZhcHBfaWQQMjIyMDM5MTc4ODIwMDg5MghjYWxsc2l0ZQEyAAEePm_7sNZksaYgEkpdFtkis9SVLAIiYmxHTAHxu85pE5Uovk4rtEhZHoe_wQQ_aem_qQSj2SVJu3zioTvsVhM0cw)  

Original Data:
Curtis et al., Nature, 2012
Pereira et al., Nature Communications, 2016

---

## 📦 Requirements
Python 3.11+

Install required packages via:

```Bash

!pip install numpy pandas scipy scikit-learn xgboost lifelines matplotlib seaborn
```

## 🚀 Getting Started
```Bash

git clone https://github.com/agnivibes/copula-risk-fusion-genomics-ml.git
cd copula-risk-fusion-genomics-ml

```

## Run the full analysis:
```bash
python copula-risk-fusion-genomics-ml.py
```
## 🔬 Research Paper
Aich, A., Hewage, S., Murshed, M. (2025). Copula Based Fusion of Clinical and Genomic Machine Learning Risk Scores for Breast Cancer Risk Stratification. [Manuscript under review]

## 📊 Citation
If you use this code or method in your own work, please cite:

@article{Aich2025A2CopulaDiabetes,
  title   = {Copula Based Fusion of Clinical and Genomic Machine Learning Risk Scores for Breast Cancer Risk Stratification},
  author  = {Aich, Agnideep and Hewage, Sameera and Murshed, Md Monzur},
  journal = {},
  year    = {2025},
  note    = {Manuscript under review}
}

## 📬 Contact
For questions or collaborations, feel free to contact:

Agnideep Aich,
Department of Mathematics, University of Louisiana at Lafayette
📧 agnideep.aich1@louisiana.edu

## 📝 License

This project is licensed under the [MIT License](LICENSE).
