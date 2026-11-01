# Sentiment Analysis of Indonesian Mobile Banking App Reviews

A reproducible pipeline that collects Google Play reviews of four Indonesian mobile banking apps (BRImo, myBCA, Livin' by Mandiri, wondr by BNI), fine-tunes IndoBERT for three-class sentiment (negative, neutral, positive), and analyzes the sentiment of all 134,910 reviews posted between 1 January and 31 August 2026.

## Main findings

- IndoBERT reached a macro-F1 of 0.944 on a held-out random test sample of 500 reviews (95% interval 0.820–0.979), and 0.897 in cross-validation, ahead of a TF-IDF and logistic regression baseline (0.861). On test reviews whose text does not occur in the training pool, macro-F1 was 0.826.
- Livin' and myBCA are predominantly negative (corrected shares 0.70 and 0.64), BRImo and wondr predominantly positive (0.29 and 0.25 negative).
- The sharp rise in negative Livin' reviews in August 2026 is largely a protest campaign about account blocking: an estimated 59–71% of negative reviews that month concern bank policy, against 2% in January–July, while the number of technical complaints did not change noticeably.
- There is no consistent evidence in any of the apps that sentiment improves with newer app versions.

Details, uncertainty intervals and limitations are in [RESULTS.md](RESULTS.md).

## Repository contents

| File | Content |
|---|---|
| `mbanking_review_pipeline.ipynb` | Data collection, audit, text normalization, sampling for labeling, labeling guidelines, label validation, fold assignment |
| `mbanking_modelling.ipynb` | Baseline, IndoBERT cross-validation, model comparison, test evaluation, continued pretraining experiment, population prediction |
| `mbanking_analysis.ipynb` | Sentiment by app, month and app version, misclassification correction, Livin' blocking/boycott cluster |
| `RESULTS.md` | Results and limitations |

The notebooks create the following folders at run time (on Google Drive when run in Colab):

```
mbanking_sentiment/
  data/raw/         scraped reviews and scrape logs
  data/interim/     study population
  data/labeling/    labeling sheets and sample keys
  data/processed/   normalized population, training pool, test set, population predictions
  results/          cross-validation and test predictions, analysis tables and figures
  models/           saved fold models
```

## Pipeline

1. **Collection.** Reviews are scraped newest-first with retries, a coverage check against the study period, and per-app logs. Timestamps are converted to Asia/Jakarta time.
2. **Audit and normalization.** Duplicates, missing text, monthly volume anomalies and text quality are reported as flags. Three text versions are produced: raw, lightly normalized, and normalized with an abbreviation dictionary.
3. **Sampling and labeling.** A random test sample and a training pool with quotas per app and month are drawn. The labeling sheet hides rating, date and split. The labeling rules and open decisions are in the pipeline notebook. A second labeling round used the first model to select informative reviews for the training pool only.
4. **Validation.** Label consistency across identical texts, rating-label discordance, class balance, and agreement with a human annotator on a subset are checked before training.
5. **Modelling.** `StratifiedGroupKFold` folds keep identical texts together. IndoBERT (`indobenchmark/indobert-base-p1`) is fine-tuned with class-weighted cross-entropy; predictions of fold models and seeds are averaged. Comparisons use pooled out-of-fold predictions with bootstrap intervals that resample text groups. The test set is evaluated once.
6. **Analysis.** Predicted shares are corrected for the classifier's error rates estimated on the test set. The analysis covers apps, months, app versions, and a manually labeled topic sample for the Livin' August reviews.

## Data and privacy

Raw review data, labeled data and model predictions are not distributed in this repository. Review texts are public on Google Play but some contain personal information, such as names and addresses, and the use of scraped data is subject to the terms of the source platform. The scraper collects only review text, rating, date, app version and the number of helpful votes. User names and avatars are not stored.

## Reproducing the analysis

**Environment.** Python 3.10 or later with `google-play-scraper`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`, `matplotlib`, `torch` and `transformers`. The notebooks were developed for Google Colab with a GPU runtime; fine-tuning on CPU is possible but slow.

**Run order.** `mbanking_review_pipeline.ipynb` (Stages 1 to 4; Stages 6 and 7 after the labeling sheet has been completed), then `mbanking_modelling.ipynb`, then `mbanking_analysis.ipynb`. Each notebook reads its settings from the first code cells (period, sample sizes, seeds, number of epochs, which runs to save) and prints `FLAG` lines for anomalies it finds.

**Expected variation.** The scraped population depends on the day of collection because reviews can be edited or removed, so counts will not match exactly after a new scrape. Raw files are cached, and a finished scrape is not repeated. Fine-tuning results vary by about 0.02 macro-F1 between seeds, which the notebooks account for by averaging three seeds and reporting intervals.

## Labeling

Reviews are labeled from the text alone as negative (0), neutral (1) or positive (2). The guidelines cover mixed reviews, requests and suggestions, situational questions, sarcasm and short approvals. Labels were assigned with an LLM following these guidelines, and a human annotator labeled 190 reviews to estimate agreement (Cohen's kappa 0.88 on confident reviews and 0.33 on borderline ones).

## Limitations

Main limitations are the small neutral class, label conventions for borderline reviews, assumptions behind the misclassification correction, incomplete app version information, and the fact that all comparisons describe associations. See [RESULTS.md](RESULTS.md) for the full list.

## Acknowledgements

IndoBERT is the `indobenchmark/indobert-base-p1` model from the IndoNLU benchmark:

Wilie, B., Vincentio, K., Winata, G. I., et al. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding.* Proceedings of AACL-IJCNLP 2020.

This project is independent of the banks whose apps are analyzed. Its findings describe the reviews collected in the study period and are not an assessment of the banks or their services.
