# Informfully Recommenders

![Informfully](https://raw.githubusercontent.com/Informfully/Documentation/main/docs/source/img/logo_banner.png)

Welcome to [Informfully](https://informfully.ch/)!
Informfully is an open-source reproducibility platform for content distribution and user experiments.

To view the full documentation, please visit [Informfully at Read the Docs](https://informfully.readthedocs.io/).
It is the combined documentation for all [code repositories](https://github.com/orgs/Informfully/repositories).

**Links and Resources:** [GitHub](https://github.com/orgs/Informfully) | [Website](https://informfully.ch) | [X](https://x.com/informfully) | [Documentation](https://informfully.readthedocs.io) | [DDIS@UZH](https://www.ifi.uzh.ch/en/ddis.html) | [Google Play](https://play.google.com/store/apps/details?id=ch.uzh.ifi.news) | [App Store](https://apps.apple.com/us/app/informfully/id1460234202)

RecSys '24 Challenge
--------------------

This repository is part of a RecSys'24 Challenge submission.
As we are unable to redistribute any of the challenge datasets, please go and download the required datasets at: https://recsys.eb.dk/#dataset

To calculate the article ranking shared in the prediction folder, please run `sample.py` located in the random walk folder.
Structure the input data to match the following format:

```python
  # List of articles to populate the graph with article nodes.
  # The labels can be anything as long as the are unique identifiers.
  articleCollection = ["A", "B", "C"]

  # List of users to populate the graph with user nodes.
  # Scoring is disabled in the current setup (all users share the same score).
  userScores = [1, 1, 1]

  # List of user-item interactions to create edges between user and item nodes.
  # The position in the userHistory array indicates the user in userScores.
  # E.g., the third user in userScores read articles B and C.
  userHistory = [["A"], ["B"], ["B","C"]]
```

<!--

 ## Algorithms

| Algorithm               | Source                  |
| ----------------------- | ----------------------- |
| Political Diversity     | TBD                     |
| Deliberative Diversity  | TBD                     |
| RP3B Random Walk        | TBD                     |

 -->

## Citation

If you use any code or data of this repository in a scientific publication, we ask you to cite the following papers:

<!--Update once the final version of the paper has been published.-->

- [Recommendations for the Recommenders: Reflections on
  Prioritizing Diversity in the RecSys Challenge](https://www.researchgate.net/publication/383261868_Recommendations_for_the_Recommenders_Reflections_on_Prioritizing_Diversity_in_the_RecSys_Challenge), Heitz *et al.*, Proceedings of the Recommender Systems Challenge 2024, 2024.

  ```
  @inproceedings{heitz2024recommendations,
    title={Recommendations for the Recommenders: Reflections on
  Prioritizing Diversity in the RecSys Challenge},
    author={Heitz, Lucien and Inel, Oana and Vrijenhoek, Sanne},
    booktitle={Proceedings of the Recommender Systems Challenge 2024},
    year={2024}
  }
  ```

- [Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications](https://dl.acm.org/doi/abs/10.1145/2955101), Paudel *et al.*, ACM Transactions on Interactive Intelligent Systems, 2016.

  ```
  @article{paudel2016updatable,
    title={Updatable, accurate, diverse, and scalable recommendations for interactive applications},
    author={Paudel, Bibek and Christoffel, Fabian and Newell, Chris and Bernstein, Abraham},
    journal={ACM Transactions on Interactive Intelligent Systems (TiiS)},
    volume={7},
    number={1},
    pages={1--34},
    year={2016},
    publisher={ACM New York, NY, USA}
  }
  ```

  - [Informfully - Research Platform for Reproducible User Studies](https://www.researchgate.net/publication/383261885_Informfully_-_Research_Platform_for_Reproducible_User_Studies), Heitz *et al.*, Proceedings of the 18th ACM Conference on Recommender Systems, 2024.

  ```
  @inproceedings{heitz2024informfully,
    title={Informfully - Research Platform for Reproducible User Studies},
    author={Heitz, Lucien and Croci, Julian A and Sachdeva, Madhav and Bernstein, Abraham},
    booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
    year={2024}
  }
  ```

## Contributing

Your are welcome to contribute to the Informfully ecosystem and become a part of our community. Feel free to:
  - fork any of the [Informfully repositories](https://github.com/Informfully)
  - join and write on the [dicussion board](https://github.com/orgs/Informfully/discussions)
  - make changes and create pull requests

Please post your feature requests and bug reports in our [GitHub issues](https://github.com/Informfully/Documentation/issues) section.

![Screenshots](https://raw.githubusercontent.com/Informfully/Documentation/main/docs/source/img/app_screens.png)
