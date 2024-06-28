# Informfully Recommender

![Informfully](https://raw.githubusercontent.com/Informfully/Documentation/main/docs/source/img/logo_banner.png)

Welcome to [Informfully](https://informfully.ch/)!
Informfully is a open-source reproducibility platform for content distribution and user experiments.

To view the full documentation, please visit [Informfully at Read the Docs](https://informfully.readthedocs.io/).
It is the combined documentation for all [code repositories](https://github.com/orgs/Informfully/repositories).

**Links and Resources:** [Website](https://informfully.ch/) | [Documentation](https://informfully.readthedocs.io/) | [GitHub](https://github.com/orgs/Informfully/repositories) | [DDIS@UZH](https://www.ifi.uzh.ch/en/ddis.html) | [Google Play](https://play.google.com/store/apps/details?id=ch.uzh.ifi.news) | [App Store](https://apps.apple.com/us/app/informfully/id1460234202)

## Instructions
This repository is part of a RecSys'24 Challenge submission.
As we are unable to redistribute any of the challenge datasets, please go and download it at: https://recsys.eb.dk/#dataset

To calculate the article predictions, please `sample.py` located in the random walk folder.
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
  # E.g., the thirs user in userScores read article B and C.
  userHistory = [["A"], ["B"], ["B","C"]]
```

## Citation
If you use any Informfully code/repository in a scientific publication, we ask you to cite the following papers:

- [Deliberative Diversity for News Recommendations - Operationalization and Experimental User Study](https://dl.acm.org/doi/10.1145/3604915.3608834), Heitz *et al.*, Proceedings of the 17th ACM Conference on Recommender Systems, 813–819, 2023.

  ```
  @inproceedings{heitz2023deliberative,
    title={Deliberative Diversity for News Recommendations: Operationalization and Experimental User Study},
    author={Heitz, Lucien and Lischka, Juliane A and Abdullah, Rana and Laugwitz, Laura and Meyer, Hendrik and Bernstein, Abraham},
    booktitle={Proceedings of the 17th ACM Conference on Recommender Systems},
    pages={813--819},
    year={2023}
  }
  ```

- [Benefits of Diverse News Recommendations for Democracy: A User Study](https://www.tandfonline.com/doi/full/10.1080/21670811.2021.2021804), Heitz *et al.*, Digital Journalism, 10(10): 1710–1730, 2022.

  ```
  @article{heitz2022benefits,
    title={Benefits of diverse news recommendations for democracy: A user study},
    author={Heitz, Lucien and Lischka, Juliane A and Birrer, Alena and Paudel, Bibek and Tolmeijer, Suzanne and Laugwitz, Laura and Bernstein, Abraham},
    journal={Digital Journalism},
    volume={10},
    number={10},
    pages={1710--1730},
    year={2022},
    publisher={Taylor \& Francis}
  }
  ```

## Contributing
Your are welcome to contribute to the Informfully ecosystem and become a part of your cummunity. Feel free to:
  - fork any of the [Informfully repositories](https://github.com/Informfully/Documentation) and
  - make changes and create pull requests.

Please post your feature requests and bug reports in our [GitHub issues](https://github.com/Informfully/Documentation/issues) section.

## License
Released under the [MIT License](LICENSE). (Please note that the respective copyright licenses of third-party libraries and dependencies apply.)

![Screenshots](https://raw.githubusercontent.com/Informfully/Documentation/main/docs/source/img/app_screens.png)
