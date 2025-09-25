# Informfully Recommenders

![Informfully](https://raw.githubusercontent.com/Informfully/Documentation/main/docs/source/img/logo_banner.png)

Welcome to Informfully ([GitHub](https://github.com/orgs/Informfully) & [Website](https://informfully.ch/))!
Informfully is an open-source reproducibility platform for content distribution and user experiments.

To view the full documentation, please visit [Informfully at Read the Docs](https://informfully.readthedocs.io/).
It is the combined documentation for all [code repositories](https://github.com/orgs/Informfully/repositories).

**Links and Resources:** [GitHub](https://github.com/orgs/Informfully) | [Website](https://informfully.ch) | [X](https://x.com/informfully) | [Documentation](https://informfully.readthedocs.io) | [DDIS@UZH](https://www.ifi.uzh.ch/en/ddis.html) | [Google Play](https://play.google.com/store/apps/details?id=ch.uzh.ifi.news) | [App Store](https://apps.apple.com/us/app/informfully/id1460234202)

## Pipeline Overview

Informfully Recommenders is a norm-aware extension of [Cornac](https://github.com/PreferredAI/cornac).
Please see the [Experiments Repository](https://github.com/Informfully/Experiments) for an overview of our past offline and online studies using this framework as back end.
And see the [Online Tutorial](https://github.com/Informfully/Experiments/tree/main/experiments/tutorial) for a quick introduction on how to use this repository.

![Informfully Recommenders Pipeline Overview](https://raw.githubusercontent.com/Informfully/Documentation/refs/heads/main/docs/source/uml/framework_extension_v4.2.png)

Please find below an overview of the norm-aware Extension of datasets, models, re-rankers, and metrics for which Informfully Recommenders provide out-of-the-box support.
Please note that this repository is fully backward compatible.
It includes and supports all elements that were already part of Cornac.

### Pre-processing Stage

| [Dataset](https://informfully.readthedocs.io/en/latest/loading.html) | Language and Type | Links |
|-|-|-|
| EB-NeRD | Danish News Dataset | [Website](https://recsys.eb.dk) |
| MIND | English (US) News Dataset | [Website](https://msnews.github.io) |
| NeMig | German News Dataset | [Website](https://github.com/andreeaiana/nemig) |

| [Augmentation](https://informfully.readthedocs.io/en/latest/augmentation.html) | Links |
|-|-|
| Sentiment Analysis | [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/augmentation/sentiment.py) |
| Named Entities | [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/augmentation/enrich_ne.py) |
| Political Actors | [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/augmentation/party.py) |
| Text Complexity | [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/augmentation/readability.py) |
| Story Cluster | [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/augmentation/story.py) |
| Article Category | [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/augmentation/category.py) |

| [Splitting](https://informfully.readthedocs.io/en/latest/splitting.html) |
|-|
| Attribute-based Sorting |
| Diversity-based Subset Construction |
| Attribute-based Stratified Splitting |
| Diversity-based Stratified Splitting |
| Clustering-based Stratified Splitting |

### In-processing Stage

| Diversity Algorithms | Description | Links |
|-|-|-|
| [PLD](https://informfully.readthedocs.io/en/latest/participatory.html) | Participatory Diversity | [Paper](https://www.tandfonline.com/doi/full/10.1080/21670811.2021.2021804), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/pld) |
| [EPD](https://informfully.readthedocs.io/en/latest/deliberative.html) | Deliberative Diversity  | [Paper](https://dl.acm.org/doi/abs/10.1145/3604915.3608834), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/epd) |

| [Random Walks](https://informfully.readthedocs.io/en/latest/randomwalks.html) | Description | Links |
|-|-|-|
| D-RDW | Diversity-Driven Random Walks | [Paper](https://doi.org/10.1145/3705328.3748016), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/drdw) |
| RP3-β | Random Walks | [Paper](https://dl.acm.org/doi/abs/10.1145/2792838.2800180), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/rp3_beta) |
| RWE-D | Random Walks with Erasure | [Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449970), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/rwe_d) |

| [Neural Model](https://informfully.readthedocs.io/en/latest/neural.html) | Description | Links |
|-|-|-|
| EMNF | Neural Baseline | [Paper](https://dl.acm.org/doi/abs/10.1145/3373807), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/enmf) |
| LSTUR | Neural Baseline | [Paper](https://aclanthology.org/P19-1033), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/lstur) |
| NPA | Neural Baseline | [Paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330665), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/npa) |
| NRMS | Neural Baseline | [Paper](https://aclanthology.org/D19-1671), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/nrms) |
| VAE | Neural Baseline | [Paper](https://dl.acm.org/doi/abs/10.1145/3178876.3186150), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/models/dae) |

### Post-processing Stage

| [Re-ranker](https://informfully.readthedocs.io/en/latest/reranker.html) | Description | Links |
|-|-|-|
| G-KL | Greedy Kullback-Leibler Divergence | [Paper](https://github.com/Informfully/Recommenders/tree/main/cornac/rerankers/greedy_kl), [Code](https://github.com/Informfully/Recommenders/blob/main/cornac/metrics/diversity.py) |
| PM-2 | Diversity by Proportionality | [Paper](https://dl.acm.org/doi/abs/10.1145/2348283.2348296), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/rerankers/pm2) |
| MMR | Maximal Marginal Relevance | [Paper](https://dl.acm.org/doi/pdf/10.1145/290941.291025), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/rerankers/mmr) |
| DAP | Dynamic Attribute Penalization | [Paper](https://doi.org/10.1145/3705328.3748016), [Code](https://github.com/Informfully/Recommenders/tree/main/cornac/rerankers/dynamic_attribute_penalization) |

| [Simulator](https://informfully.readthedocs.io/en/latest/simulator.html) | Links |
|-|-|
| Rank-based User Simulator | [Paper](https://doi.org/10.1145/3705328.3748016), [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/rerankers/user_simulator.py) |
| Preference-based User Simulator | [Paper](https://doi.org/10.1145/3705328.3748016), [Script](https://github.com/Informfully/Recommenders/blob/main/cornac/rerankers/user_simulator.py) |

### Evaluation Stage

| [Metric](https://informfully.readthedocs.io/en/latest/metrics.html) | Description | Links |
|-|-|-|
| Gini | Gini Coefficinet | [Paper](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_16), [Code](https://github.com/Informfully/Recommenders/blob/main/cornac/metrics/diversity.py) |
| ILD | Intra-list Distance | [Paper](https://api.semanticscholar.org/CorpusID:11075976), [Code](https://github.com/Informfully/Recommenders/blob/main/cornac/metrics/diversity.py) |
| RADio | RADio Divergence | [Paper](https://dl.acm.org/doi/abs/10.1145/3523227.3546780), [Code](https://github.com/Informfully/Recommenders/blob/main/cornac/metrics/diversity.py) |

| Gini Dimension | Required Augmentation |
|-|-|
| Category Gini | Article Category |
| Sentiment Gini | Sentiment Analysis |
| Party Gini | Political Actors |

| ILD Dimension | Required Augmentation |
|-|-|
| Category ILD | Article Category |
| Sentiment ILD | Sentiment Analysis |
| Party ILD | Political Actors |

| RADio Dimension | Required Augmentation |
|-|-|
| Activation | Sentiment Analysis |
| Calibration | Article Category, Text Complexity |
| Fragmentation | Story Cluster |
| Alternative Voices | Political Actors, Named Entities |
| Representation | Political Actors, Named Entities |

| Sample Scripts | Links |
|-|-|
| Accuracy Evaluation (AUC) | [Script](https://github.com/Informfully/Experiments/tree/main/experiments/recsys_2025/evaluation_scripts/check_accuracy) |
| Traditional Diversity Evaluation (Gini and ILD) | [Script](https://github.com/Informfully/Experiments/blob/main/experiments/recsys_2025/evaluation_scripts/check_diversity/check_diversity.py) |
| Normative Diversity Evaluation (RADio) | [Script](https://github.com/Informfully/Experiments/tree/main/experiments/recsys_2025/evaluation_scripts/check_ntd) |

Item visualization is done using the [Informfully Platform](https://github.com/Informfully/Platform).
Please look at the relevant documentation page for a [demo script](https://informfully.readthedocs.io/en/latest/recommendations.html).

## Citation

If you use any code or data from this repository in a scientific publication, we ask you to cite the following papers:

- [Informfully Recommenders – A Reproducibility Framework for Diversity-aware Intra-session Recommendations](https://doi.org/10.1145/3705328.3748148), Heitz *et al.*, Proceedings of the 19th ACM Conference on Recommender Systems, 2025.

  ```tex
  @inproceedings{heitz2025recommenders,
    title={Informfully Recommenders – A Reproducibility Framework for Diversity-aware Intra-session Recommendations},
    author={Heitz, Lucien and Li, Runze and Inel, Oana and Bernstein, Abraham},
    booktitle={Proceedings of the 19th ACM Conference on Recommender Systems},
    pages={792--801},
    year={2025},
  }
  ```
  
- [Informfully - Research Platform for Reproducible User Studies](https://doi.org/10.1145/3640457.3688066), Heitz *et al.*, Proceedings of the 18th ACM Conference on Recommender Systems, 2024.

  ```tex
  @inproceedings{heitz2024informfully,
    title={Informfully - Research Platform for Reproducible User Studies},
    author={Heitz, Lucien and Croci, Julian A and Sachdeva, Madhav and Bernstein, Abraham},
    booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
    pages={660--669},
    year={2024}
  }
  ```
  
- [Multi-Modal Recommender Systems: Hands-On Exploration](http://jmlr.org/papers/v21/19-805.html), Truong *et al.*, Proceedings of the 15th ACM Conference on Recommender Systems, 2021.

  ```tex
  @inproceedings{truong2021multi,
    title={Multi-modal recommender systems: Hands-on exploration},
    author={Truong, Quoc-Tuan and Salah, Aghiles and Lauw, Hady},
    booktitle={Fifteenth ACM Conference on Recommender Systems},
    pages={834--837},
    year={2021}
  }

## Contributing

You are welcome to contribute to the Informfully ecosystem and become a part of our community.
Feel free to:

- Fork any of the [Informfully repositories](https://github.com/Informfully/Documentation).
- Suggest new features in [Future Release](https://github.com/orgs/Informfully/projects/1).
- Make changes and create pull requests.

Please post your feature requests and bug reports in our [GitHub issues](https://github.com/Informfully/Documentation/issues) section.

## License

Released under the [Apache License 2.0](LICENSE). (Please note that the respective copyright licenses of third-party libraries and dependencies apply.)

![Screenshots](https://raw.githubusercontent.com/Informfully/Documentation/main/docs/source/img/app_screens.png)
