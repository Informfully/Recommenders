# Cornac examples directory

## Basic usage

[first_example.py](first_example.py) - Your very first example with Cornac.

[pmf_ratio.py](pmf_ratio.py) - Splitting data into train/val/test sets based on provided sizes (RatioSplit).

[given_data.py](given_data.py) - Evaluate the models with your own data splits.

[propensity_stratified_evaluation_example.py](propensity_stratified_evaluation_example.py) - Evaluate the models with [Propensity Stratified Evaluation](https://arxiv.org/abs/2104.08912) method.

[vbpr_tradesy.py](vbpr_tradesy.py) - Image features associate with items/users.

[c2pf_example.py](c2pf_example.py) - Items/users networks as graph modules.

[conv_mf_example.py](conv_mf_example.py) - Text data associate with items/users.

[param_search.py](param_search.py) - Hyper-parameter tuning with GridSearch and RandomSearch.

---

## Multimodal Algorithms (Using Auxiliary Data)

### Graph

[c2pf_example.py](c2pf_example.py) - Collaborative Context Poisson Factorization (C2PF) with Amazon Office dataset.

[cvaecf_filmtrust.py](cvaecf_filmtrust.py) - Fit and evaluate Conditional VAE (CVAECF) on the FilmTrust dataset.

[gcmc_example.py](gcmc_example.py) - Graph Convolutional Matrix Completion (GCMC) example with MovieLens 100K dataset.

[lightgcn_example.py](lightgcn_example.py) - LightGCN example with CiteULike dataset.

[gcmc_example.py](gcmc_example.py) - Graph Convolutional Matrix Completion (GCMC) example with MovieLens 100K dataset.

[lightgcn_example.py](lightgcn_example.py) - LightGCN example with CiteULike dataset.

[mcf_office.py](mcf_office.py) - Fit Matrix Co-Factorization (MCF) to the Amazon Office dataset.

[ngcf_example.py](ngcf_example.py) - NGCF example with CiteULike dataset.

[ngcf_example.py](ngcf_example.py) - NGCF example with CiteULike dataset.

[pcrl_example.py](pcrl_example.py) - Probabilistic Collaborative Representation Learning (PCRL) Amazon Office dataset.

[sbpr_epinions.py](sbpr_epinions.py) - Social Bayesian Personalized Ranking (SBPR) with Epinions dataset.

[sorec_filmtrust.py](sorec_filmtrust.py) - Social Recommendation using PMF (Sorec) with FilmTrust dataset.

### Text

[cdl_example.py](cdl_example.py) - Collaborative Deep Learning (CDL) with CiteULike dataset.

[cdr_example.py](cdr_example.py) - Collaborative Deep Ranking (CDR) with CiteULike dataset.

[companion_example.py](companion_example.py) - Comparative Aspects and Opinions Ranking for Recommendation Explanations (Companion) with Amazon Toy and Games dataset.

[companion_example.py](companion_example.py) - Comparative Aspects and Opinions Ranking for Recommendation Explanations (Companion) with Amazon Toy and Games dataset.

[conv_mf_example.py](conv_mf_example.py) - Convolutional Matrix Factorization (ConvMF) with MovieLens dataset.

[ctr_example_citeulike.py](ctr_example_citeulike.py) - Collaborative Topic Regression (CTR) with CiteULike dataset.

[cvae_example.py](cvae_example.py) - Collaborative Variational Autoencoder (CVAE) with CiteULike dataset.

[dmrl_example.py](dmrl_example.py) - Disentangled Multimodal Representation Learning (DMRL) with citeulike dataset.

[trirank_example.py](trirank_example.py) - TriRank with Amazon Toy and Games dataset.

[dmrl_example.py](dmrl_example.py) - Disentangled Multimodal Representation Learning (DMRL) with citeulike dataset.

[trirank_example.py](trirank_example.py) - TriRank with Amazon Toy and Games dataset.

[efm_example.py](efm_example.py) - Explicit Factor Model (EFM) with Amazon Toy and Games dataset.

[hft_example.py](hft_example.py) - Hidden Factor Topic (HFT) with MovieLen 1m dataset.

[lrppm_example.py](lrppm_example.py) - Learn to Rank user Preferences based on Phrase-level sentiment analysis across Multiple categories (LRPPM) with Amazon Toy and Games dataset.

[lrppm_example.py](lrppm_example.py) - Learn to Rank user Preferences based on Phrase-level sentiment analysis across Multiple categories (LRPPM) with Amazon Toy and Games dataset.

[mter_example.py](mter_example.py) - Multi-Task Explainable Recommendation (MTER) with Amazon Toy and Games dataset.

### Image

[causalrec_clothing.py](causalrec_clothing.py) - CausalRec with Clothing dataset.

[dmrl_clothes_example.py](dmrl_clothes_example.py) - Disentangled Multimodal Representation Learning (DMRL) with Amazon clothing dataset.

[dmrl_clothes_example.py](dmrl_clothes_example.py) - Disentangled Multimodal Representation Learning (DMRL) with Amazon clothing dataset.

[vbpr_tradesy.py](vbpr_tradesy.py) - Visual Bayesian Personalized Ranking (VBPR) with Tradesy dataset.

[vmf_clothing.py](vmf_clothing.py) - Visual Matrix Factorization (VMF) with Amazon Clothing dataset.

## Unimodal Algorithms

[biased_mf.py](biased_mf.py) - Matrix Factorization (MF) with biases.

[bpr_netflix.py](bpr_netflix.py) - Example to run Bayesian Personalized Ranking (BPR) with Netflix dataset.

[ease_movielens.py](ease_movielens.py) - Embarrassingly Shallow Autoencoders (EASEᴿ) with MovieLens 1M dataset.

[fm_example.py](fm_example.py) - Example to run Factorization Machines (FM) with MovieLens 100K dataset.

[hpf_movielens.py](hpf_movielens.py) - (Hierarchical) Poisson Factorization vs BPR on MovieLens data.

[ibpr_example.py](ibpr_example.py) - Example to run Indexable Bayesian Personalized Ranking.

[knn_movielens.py](knn_movielens.py) - Example to run Neighborhood-based models with MovieLens 100K dataset.

[mmmf_exp.py](mmmf_exp.py) - Maximum Margin Matrix Factorization (MMMF) with MovieLens 100K dataset.

[ncf_example.py](ncf_example.py) - Neural Collaborative Filtering (GMF, MLP, NeuMF) with Amazon Clothing dataset.

[nmf_example.py](nmf_example.py) - Non-negative Matrix Factorization (NMF) with RatioSplit.

[pmf_ratio.py](pmf_ratio.py) - Probabilistic Matrix Factorization (PMF) with RatioSplit.

[recvae_example.py](recvae_example.py) - New Variational Autoencoder for Top-N Recommendations with Implicit Feedback (RecVAE).

[recvae_example.py](recvae_example.py) - New Variational Autoencoder for Top-N Recommendations with Implicit Feedback (RecVAE).

[skm_movielens.py](skm_movielens.py) - SKMeans vs BPR on MovieLens data.

[svd_example.py](svd_example.py) - Singular Value Decomposition (SVD) with MovieLens dataset.

[vaecf_citeulike.py](vaecf_citeulike.py) - Variational Autoencoder for Collaborative Filtering (VAECF) with CiteULike dataset.

[wmf_example.py](wmf_example.py) - Weighted Matrix Factorization with CiteULike dataset.

----

## Next-Item Algorithms

[spop_yoochoose.py](spop_yoochoose.py) - Next-item recommendation based on item popularity.

[gru4rec_yoochoose.py](gru4rec_yoochoose.py) - Example of Session-based Recommendations with Recurrent Neural Networks (GRU4Rec).

----

## Next-Basket Algorithms

[gp_top_tafeng.py](gp_top_tafeng.py) - Next-basket recommendation model that merely uses item top frequency.

[dnntsp_tafeng.py](dnntsp_tafeng.py) - Predicting Temporal Sets with Deep Neural Networks (DNNTSP).

[beacon_tafeng.py](beacon_tafeng.py) - Correlation-Sensitive Next-Basket Recommendation (Beacon).

[tifuknn_tafeng.py](tifuknn_tafeng.py) - Example of Temporal-Item-Frequency-based User-KNN (TIFUKNN).

[upcf_tafeng.py](upcf_tafeng.py) - Example of Recency Aware Collaborative Filtering for Next Basket Recommendation (UPCF).

[dae_movielens.py](dae_movielens.py) - Denoising Autoencoder with Movielens dataset

---

## Experiment Scripts (Standard Sequential Workflow with Reranking)

These scripts follow a standard sequential evaluation workflow where the model is trained, reranked, and evaluated without intermediate checkpointing or modular execution.

- **[standard_dae_reranking_workflow.py](standard_dae_reranking_workflow.py)** - Implements the Multinomial Denoising Autoencoder (DAE) with reranking.
- **[standard_drdw_workflow.py](standard_drdw_workflow.py)** - Executes the Diversity-Driven Random Walk model (D-RDW).
- **[standard_mostpop_reranking_workflow.py](standard_mostpop_reranking_workflow.py)** - Applies the MostPop (Most Popular) model with reranking.
- **[standard_nrms_reranking_workflow.py](standard_nrms_reranking_workflow.py)** - Utilizes the Neural News Recommendation model with Multi-Head Self-Attention (NRMS) along with reranking.

---

## Pipeline Experiment Scripts (Flexible Modular Workflow with Reranking)

Pipeline experiment scripts enable modular experimentation by supporting flexibility to skip steps, load pre-generated recommendations, and configure the workflow via `.ini` files.

- **[pipeline_dae_reranking_workflow.py](pipeline_dae_reranking_workflow.py)** - Modular pipeline for the Multinomial Denoising Autoencoder (DAE) with reranking.
- **[pipeline_drdw_workflow.py](pipeline_drdw_workflow.py)** - Flexible pipeline experiment for the Diversity-Driven Random Walk model (D-RDW).
- **[pipeline_mostpop_reranking_workflow.py](pipeline_mostpop_reranking_workflow.py)** - Modular pipeline for the Most Popular (MostPop) model with reranking.
- **[pipeline_nrms_reranking_workflow.py](pipeline_nrms_reranking_workflow.py)** - Pipeline experiment for the Neural News Recommendation model with Multi-Head Self-Attention (NRMS) with reranking.
- **[pipeline_epd_reranking_workflow.py](pipeline_epd_reranking_workflow.py)** - Pipeline for the EPD model with reranking. The EPD model is based on the reference paper: _Deliberative Diversity for News Recommendations: Operationalization and Experimental User Study_. Note: The EPD codebase is not included here; recommendations are generated using another team's Cornac implementation.

- **[pipeline_pld_reranking_workflow.py](pipeline_pld_reranking_workflow.py)** - Pipeline experiment for the PLD model with reranking. The PLD model is based on the reference paper: _Benefits of Diverse News Recommendations for Democracy: A User Study_. Note: The PLD model is not integrated here; recommendations are generated using another team's Cornac implementation.
- **[pipeline_rdw_reranking_workflow.py](pipeline_rdw_reranking_workflow.py)** - Pipeline experiment for the RDW model with reranking. The RDW model is introduced in the reference paper: _Blockbusters and Wallflowers: Accurate, Diverse, and Scalable Recommendations with Random Walks_. Note: The RDW codebase is not included here; recommendations are generated using another team's Cornac implementation.
