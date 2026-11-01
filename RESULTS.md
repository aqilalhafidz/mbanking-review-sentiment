# Results and Limitations

This document reports the findings of the sentiment analysis of Google Play reviews for four Indonesian mobile banking apps (BRImo, myBCA, Livin' by Mandiri, wondr by BNI) between 1 January and 31 August 2026. It follows the pipeline described in the README: data collection, manual labeling, fine-tuning of IndoBERT, and analysis of the scored review population.

## 1. Data and labeling

**Population.** Reviews were collected with `google-play-scraper` (Indonesian language and country settings) at the end of October 2026 and restricted to the study period. Timestamps were converted from UTC to Asia/Jakarta. After removing duplicate review identifiers, the population contains 134,910 reviews.

| App | Reviews |
|---|---|
| BRImo | 59,506 |
| Livin' by Mandiri | 35,968 |
| wondr by BNI | 30,604 |
| myBCA | 8,832 |

**Labels.** Reviews were labeled as negative, neutral or positive from the review text alone (the star rating was hidden from the labeling sheet). The labeling rules are documented in the data pipeline notebook. Labels were assigned with an LLM (Claude, Anthropic) following the written guidelines. The project owner independently labeled 190 reviews (100 random reviews with a confident label and 90 reviews marked as uncertain) to estimate agreement: Cohen's kappa was 0.88 on the random reviews and 0.33 on the uncertain reviews. Disagreements concentrated on mixed reviews (praise followed by a complaint), requests and suggestions, and situational questions.

**Splits.**

| Set | Reviews | How it was drawn |
|---|---|---|
| Test | 500 | Simple random sample, 125 reviews per app (254 negative, 29 neutral, 217 positive) |
| Training pool, round 1 | 1,000 | 250 per app, equal quotas across months, at most two copies of an identical text |
| Training pool, round 2 | 400 | Selected with the round-1 model: 160 with the highest neutral probability, 160 with the highest predictive entropy, 80 random |

Cross-validation folds were built with `StratifiedGroupKFold`, grouping identical texts so that a text never appears in two folds. The test set was evaluated once, after the final configuration was fixed.

## 2. Classifier

### 2.1 Cross-validation

Out-of-fold predictions pooled over five folds and three seeds (1,400 training-pool reviews). Intervals are 95% bootstrap intervals that resample whole text groups.

| Model | Macro-F1 | 95% interval | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|
| IndoBERT (`indobenchmark/indobert-base-p1`) | **0.897** | 0.879–0.915 | 0.937 | 0.830 | 0.926 |
| TF-IDF + logistic regression (best text version) | 0.861 | 0.840–0.880 | 0.905 | 0.781 | 0.898 |

The paired difference in macro-F1 between the baseline and IndoBERT is −0.036 (95% interval −0.056 to −0.017). The training pool contains model-targeted reviews and a neutral share of 18%, so these figures are not estimates of performance on the population. On the 1,000 round-1 reviews alone, which were drawn without model targeting, IndoBERT reaches a macro-F1 of 0.865 (neutral F1 0.658) and the baseline 0.804 (neutral F1 0.525). The second labeling round improved the baseline (0.762 before, 0.804 after) but did not change IndoBERT on these reviews (0.863–0.870 across runs).

### 2.2 Preprocessing variants and continued pretraining

Compared in the first round (single seed, 1,000 reviews): raw text (A), light normalization (B), light normalization plus an abbreviation dictionary (C), emoji replaced by words (D), and continued masked-language-model pretraining on 81,874 unique review texts (test texts excluded) followed by fine-tuning on B. The checkpoint contains no masked-language-model head, so the head was newly initialized for pretraining.

| Variant | Macro-F1 |
|---|---|
| A, raw | 0.857 |
| B, light normalization | 0.870 |
| C, plus abbreviation dictionary | 0.840 |
| D, emoji to words | 0.844 |
| B with continued pretraining | 0.853 |

No variant differs from A beyond bootstrap uncertainty (paired intervals include zero), and seed-to-seed variation of the same configuration is about 0.02. Variant B was used.

### 2.3 Learning curve

With IndoBERT on the round-1 pool, out-of-fold macro-F1 was 0.660, 0.795, 0.831 and 0.852 for 200, 400, 600 and 800 training reviews per fold. The last increment (+0.021) is of the same size as the seed-to-seed variation.

### 2.4 Held-out test set

| View | Reviews | Accuracy | Macro-F1 | 95% interval | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|---|---|
| All test reviews | 500 | 0.974 | **0.944** | 0.820–0.979 | 0.978 | 0.873 | 0.982 |
| Text not in the training pool | 375 | 0.965 | 0.826 | 0.695–0.923 | 0.978 | 0.533 | 0.965 |
| Confident labels only | 472 | 0.989 | 0.980 | 0.865–0.998 | 0.989 | 0.958 | 0.993 |

Confusion matrix (rows: true class, columns: predicted class):

| | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 250 | 2 | 2 |
| Neutral | 3 | 24 | 2 |
| Positive | 4 | 0 | 213 |

The first view reflects the population as a whole, where about 42% of reviews are texts that occur more than once. The second view measures generalization to texts the model has not seen. The neutral class has 29 test reviews, so neutral F1 and the intervals are rough. Of the 13 misclassified test reviews, about eight are among the 28 marked as uncertain.

## 3. Sentiment by app

Shares of all reviews, corrected for the classifier's error rates measured on the test set (matrix-inversion prevalence estimation with a non-negativity constraint). Intervals resample the test set by text group and therefore reflect classifier uncertainty only, because the reviews form a census of the collected period.

| App | Negative | Neutral | Positive |
|---|---|---|---|
| Livin' by Mandiri | 0.702 (0.668–0.716) | 0.043 (0.031–0.090) | 0.255 (0.233–0.264) |
| myBCA | 0.644 (0.598–0.658) | 0.066 (0.053–0.138) | 0.290 (0.257–0.301) |
| BRImo | 0.289 (0.245–0.303) | 0.066 (0.056–0.137) | 0.645 (0.611–0.661) |
| wondr by BNI | 0.253 (0.200–0.268) | 0.079 (0.067–0.165) | 0.668 (0.627–0.685) |

Two apps are predominantly negative (Livin', myBCA) and two predominantly positive (BRImo, wondr). The ordering is the same for uncorrected predicted shares. Neutral shares have wide intervals and should not be compared across apps. The model's confidence was below 0.6 for 0.7–1.2% of reviews.

## 4. Monthly patterns and the Livin' August cluster

BRImo (0.21–0.35), wondr (0.20–0.32) and myBCA (0.55–0.72) show negative shares within stable ranges across months. Livin' is volatile: monthly volume was 1,456, 1,637, 5,372, 2,499, 1,767, 1,448, 3,326 and 18,463 reviews from January to August, and the corrected negative share ranged from 0.31 (July) to **0.92 (August)**. Livin' volume in August is 8.7 times its monthly median. Keywords for promotions and prize draws appear in at most 0.3% of reviews in any app and month, so incentivized reviews do not explain the March and July volume peaks. Those peaks remain unexplained.

**Topic of the August reviews.** A keyword pattern (account blocking, boycott, law enforcement) flagged 5,214 Livin' reviews, 28% of Livin' reviews in August and fewer than 25 reviews in each other app; 99% of flagged reviews were predicted negative. A manual check of 15 random flagged reviews found 14 relevant, but reviews that mention blocking in policy terms without the pattern's keywords were missed, so the pattern has limited recall. A more reliable estimate came from labeling the topic of 400 randomly drawn Livin' reviews predicted as negative (300 from August, 100 from January to July).

| Topic (share of predicted-negative Livin' reviews) | January–July (n = 100) | August (n = 300) |
|---|---|---|
| Bank policy and boycott (account blocking, protest) | 2% (0.6–7%) | **59%** (54–65%) |
| Bank policy, broad definition (adds generic insults directed at the bank) | 2% | 71% (66–76%) |
| App technical problems (login, verification, crashes, slowness) | 87% | 7% |
| Generic or other | 3% | 29% |
| Service and transactions | 8% | 5% |

Derived estimates (uncorrected predicted labels):

- An estimated 9,900 (narrow definition) to 11,900 (broad definition) August reviews concern bank policy.
- Removing them, the negative share of Livin' in August falls from 0.91 to 0.80 (narrow) or 0.74 (broad); in January–July it stays at 0.48. If all generic insults are also removed, the August share is about 0.53. The August share without the campaign therefore lies between about 0.53 and 0.80, depending on how generic insults are treated.
- Over the whole period, the Livin' negative share is about 0.55–0.58 without policy reviews, compared with 0.70 with all reviews. Livin' remains more negative than BRImo and wondr.
- The count of technical complaints is approximately unchanged: about 1,050 per month from January to July against an estimated 1,100 in August (roughly 740–1,700 given sampling uncertainty). The August surge is therefore not evidence of a decline in app quality.

## 5. App versions

Version information is incomplete: 18% of BRImo reviews, 36% of Livin' reviews, 22% of myBCA reviews and 10% of wondr reviews have no version. Reviews without a version are more often negative (predicted negative share 54% versus 24% for wondr, 80% versus 64% for Livin', 37% versus 29% for BRImo, 68% versus 64% for myBCA), and in Livin' only 51% of August reviews carry a version, against 75–81% in other months. The version is the version installed on the reviewer's device, not the version current at the time of the review: old versions keep receiving reviews until the end of the period, and reviewers on very old versions are a self-selected group (for example, users who cannot update).

Selected observations among versions with at least 100 reviews:

- **Livin':** the negative share fell from 0.59 (3.3.0) to 0.36 (3.6.0) and then rose to 0.85 for version 3.7.0 (7,895 reviews, 34% of Livin' reviews with a version; first review on 15 July). In the August topic sample, 57% of 3.7.0 reviews (61 of 108) concern bank policy and 9% (10 of 108) technical problems, similar to reviews without a version (59% policy) and to other versions (69% policy; 32 reviews). The high negative share of 3.7.0 is therefore largely attributable to the campaign and not to the version itself.
- **myBCA:** 2.15.1 (0.75; 1,444 reviews) and 2.17.1 (0.70; 898 reviews) are well above neighbouring versions.
- **BRImo:** the share fell to 0.22 for 2.95.1 and rose to 0.39 for 2.96.0.
- **wondr:** the share rose slightly from 0.22 (1.4.4) to 0.28 (1.6.0) over four versions.

A logistic regression of the predicted negative label on version order with month fixed effects gives odds ratios per version step of 0.97 (0.969–0.976) for BRImo, 0.99 (0.976–0.999) for Livin', 1.01 (0.995–1.018) for myBCA and 1.01 (0.957–1.058) for wondr. Version order and calendar month are correlated at 0.23–0.83 (0.54–0.94 when restricted to versions with at least 100 reviews), so version and time cannot be separated well. There is no consistent evidence that sentiment improves with newer versions in any app.

## 6. Validity checks

- The mean star rating decreases from positive to negative predictions: 4.87 (positive), 4.56 (neutral) and 1.86 (negative). The neutral class contains many short approvals such as "ok", which usually carry five stars.
- 6,548 reviews are predicted negative with five stars (18.7% of BRImo's predicted-negative reviews, 15.2% for wondr, 8.6% for myBCA, 5.7% for Livin'). In a random sample of 20, about 14 were genuine complaints with a high rating, about 3 were model errors and the rest ambiguous. Ratings alone would therefore misclassify a substantial share of complaints.
- 1,153 reviews are predicted positive with one or two stars (8.2% of Livin's positive predictions, about 1% elsewhere). In a random sample of 20, about a third were model errors on short slang, misspelled or emoji-only reviews, and the others were positive text with a protest rating or ambiguous text. This stratum is under 1% of the population.
- 1,137 reviews (0.84%) consist only of emoji, and all are predicted positive. A lookup of the emoji classifies 904 as positive, 65 as negative and 168 as unclear, so the number of certain errors is small.

## 7. Limitations

- **Labels.** Labels were assigned with an LLM following project-specific guidelines, with a partial human agreement check on 190 reviews. Agreement is high for clear reviews and low for borderline ones (mixed reviews, requests, situational questions). A different convention for these cases would move some reviews between classes.
- **Neutral class.** The neutral class is small (29 test reviews, about 4–5% of reviews) and heterogeneous (short approvals, trial statements, feature requests, hopes). Its F1, error rates and corrected shares are the least reliable numbers in this analysis.
- **Test set and correction.** Corrected shares assume that the error rates measured on one 500-review test sample apply to the whole population, to all apps and to all months. Intervals do not include sampling variation in the population itself or the label convention.
- **Model errors.** Errors concentrate on short reviews with slang, misspellings, sarcasm and emoji, and on mixed reviews whose convention has few training examples.
- **Training pool.** The second labeling round was selected with the first model, so out-of-fold scores on the 1,400-review pool are not estimates of population performance.
- **Review dates and reviewers.** Google Play may display the date of the latest edit instead of the original posting date. Reviewers are not a random sample of users, and organized campaigns are over-represented in some periods.
- **Topic analysis.** The topic shares for Livin' are based on 400 manually labeled reviews (300 for August), are limited to reviews predicted negative, and the treatment of generic insults is a judgment call. Intervals for the August policy share are wide (54–65%).
- **App versions.** Many reviews carry no version, missing versions are related to sentiment, the version is the installed version, and version is strongly correlated with time. Version comparisons are associations and not effects of updates.
- **Scope.** All comparisons between apps or versions describe associations in the collected reviews. They do not identify causes, and the findings apply to the study period.
