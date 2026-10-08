# Text-Based Prediction of Book Review Popularity

**Goodreads Poetry subset · Machine Learning Mini Project**

**Team:** Guntupalli Manaswi (PES1UG24CS548) and Vainik (PES1UG24CS513)  
**Course:** Semester 5, Section I · PES University

## Project overview

Why do some book reviews receive more attention than others? We classify a review as **popular** when its votes plus comments account for **more than 2%** of the total votes plus comments recorded for reviews of the same book. The label measures a review's *relative engagement* in a Goodreads snapshot; it is neither a sentiment label nor a forecast of future likes.

This is a Poetry-only, independently runnable adaptation inspired by Bridget Daly's [Stanford STATS229 project](https://github.com/bridgetdaly/goodreads_ML). We compare reviewer/book metadata, linguistic features and review text using logistic regression, XGBoost and a small neural network. The Poetry dataset, book-level split and several model settings differ from the original study, so our numbers are **not a full reproduction of the paper**.

## Repository contents

| Path | Contents |
| --- | --- |
| [`ML_mini_project_548_513.ipynb`](ML_mini_project_548_513.ipynb) | Executable Colab notebook with preprocessing, 18 experiments, evaluation, explanations and a text-only demo |
| [`Book_Review_Popularity_548_513.pdf`](Book_Review_Popularity_548_513.pdf) | Presentation exported as a PDF |
| [`results/`](results/) | Saved evaluation tables, figures, run metadata and selected model artifacts from one execution |
| [`results/requirements.txt`](results/requirements.txt) | Tested core package versions; the notebook installs its own dependencies |

The two compressed Goodreads inputs are **not stored in this repository**. In particular, downloading this repository alone does not provide the dataset or fastText language model needed for a fresh run.

## Dataset and Colab run

1. Download the two Poetry files from the dataset authors' [Goodreads dataset page](https://mengtingwan.github.io/data/goodreads.html): `goodreads_reviews_poetry.json.gz` and `goodreads_books_poetry.json.gz`. Keep both files compressed.
2. Open [`ML_mini_project_548_513.ipynb`](ML_mini_project_548_513.ipynb) in Google Colab, preferably with a fresh CPU runtime, and choose **Runtime → Run all**.
3. Select **both** `.json.gz` files when the notebook's upload picker appears. Colab may append a suffix such as `(1)` to a duplicate filename; the notebook handles this.
4. Stay connected while it runs. The notebook installs packages, downloads the official fastText `lid.176.bin` language model and NLTK resources, trains/evaluates the models, and downloads a results ZIP. To keep a copy of the execution, download the executed `.ipynb` from Colab.

The original dataset files used for the recorded run had these SHA-256 checksums (also stored in [`results/run_summary.json`](results/run_summary.json)):

| Input | SHA-256 |
| --- | --- |
| Reviews | `4bf8f720bccf3c96caddda9eeb4b6bc7b4f7ecd72ec7f9345d8c9c28eeddd781` |
| Books | `84a8c21273d28b7df29dc7f996b15f2b39e92e3c8808cd7288578c704f07c4a9` |

If Colab resets, rerun the notebook and upload the two datasets again. Package versions and seeds are recorded for reproducibility, but exact numeric results may vary across environments.

## Method

- **Filtering and target:** Remove invalid/zero-engagement reviews; require at least 10 book reviews in the metadata, at least 60 total votes plus comments for the book, and English language confidence of at least 0.90 using fastText. Compute each review's share against its book's engagement total. Label it popular when the share is **strictly greater than 0.02**.
- **Four feature sets:** A uses four metadata variables; B adds eight language features; C adds Bag of Words; D adds TF-IDF. Text vocabularies have at most 2,000 terms. Vocabulary and numeric scaling are fitted using training rows.
- **Evaluation:** Split by **book** into train, validation and test groups so a book cannot appear across groups. Compare logistic regression, XGBoost and a small scikit-learn MLP. Full and balanced training are compared for A/B; balanced training is used for C/D, giving **18 experiments**. Tune and select with validation ROC-AUC, then report test metrics.
- **Analysis:** Export model comparisons, logistic-regression coefficients, XGBoost split-count importance, sampled neural-network Kernel SHAP, correlations with engagement share, and TP/FP/FN/TN feature distributions. SHAP on the text models is sampled and exploratory.

Direct votes, comments, engagement totals and engagement share are used to construct the label and are **excluded from model inputs**. Some allowed features, such as a user's review activity in this snapshot and review age relative to the snapshot, are retrospective; this study does not claim a publication-time forecast.

## Recorded results

The following values are from the saved execution in [`results/run_summary.json`](results/run_summary.json):

| Measure | Value |
| --- | ---: |
| Raw Poetry reviews | 154,555 |
| Reviews after filtering | 9,663 |
| Books / users after filtering | 462 / 6,336 |
| Popular reviews | 2,604 (26.95%) |
| Model chosen on validation | XGBoost B, full training |
| Test ROC-AUC / accuracy | 0.768 / 0.753 |
| Test recall / specificity | 0.398 / 0.916 |
| Majority-class baseline accuracy / ROC-AUC | 0.686 / 0.500 |

Balancing the XGBoost B training data increased test recall to 0.735, with accuracy 0.678 and ROC-AUC 0.766. A different experiment, balanced XGBoost D, had a test ROC-AUC of 0.774, but **was not selected using the test set**; the selected model had the stronger validation result.

The notebook also includes `predict_review("Your review text here")` for a live example. That function uses a **separate text-only TF-IDF + logistic regression model** (test ROC-AUC 0.695). A fresh text string does not contain the historical metadata needed by the selected XGBoost B model.

## Scope and attribution

The original paper studied a much larger, multi-genre Goodreads collection. Our Poetry-only results cannot establish general performance across genres or causal effects of writing choices on engagement. The label depends on recorded book-level engagement, and reviews with zero engagement were excluded.

The project adapts ideas from [Bridget Daly's reference repository](https://github.com/bridgetdaly/goodreads_ML), inspected at commit `5297102c0dd9cb93145fd0b1fdf3bf63532bab6a`. The Goodreads files come from the [UCSD Book Graph dataset](https://mengtingwan.github.io/data/goodreads.html). The notebook and documentation were prepared with AI assistance and require the team to understand, review and explain the implementation. This repository is an independent student adaptation, not an official Stanford implementation.
