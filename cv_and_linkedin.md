# CV bullets and LinkedIn post: xG model project

All figures below come from the executed notebook (`xg_model.ipynb`) and `results/`.

## CV bullet points (choose 2-3)

**Expected Goals (xG) Model | Python, LightGBM, scikit-learn, SHAP, mplsoccer** · github.com/anthony07-max/football-xg-model

- Built an expected goals model on 57,664 non-penalty shots from 2,351 matches of StatsBomb event data. I engineered geometry, assist-type and freeze-frame features (defenders in the shooting lane, goalkeeper position), cutting log loss by 10% compared with a distance-and-angle baseline.
- Matched StatsBomb's professional xG model on 471 unseen matches (log loss 0.2584 vs 0.2577; ROC AUC 0.809 vs 0.811; difference not statistically significant on a match-level bootstrap), with better aggregate calibration (1,157 xG vs 1,159 actual goals, compared with 1,084 from StatsBomb).
- Turned the model into scouting insight: ranked 183 players' finishing using out-of-sample xG and z-scores to separate skill from luck, and produced shot maps and SHAP explanations for non-technical audiences.

*Shorter one-line version:* Built an xG model (LightGBM) on 57,664 StatsBomb shots that matches StatsBomb's own xG on unseen matches (log loss 0.2584 vs 0.2577, AUC 0.809 vs 0.811), with better aggregate calibration.

## LinkedIn post

Can a model built on free public data keep up with a professional expected goals (xG) model? ⚽📊

For my latest football analytics project, I built an xG model from scratch using StatsBomb's open data: 57,664 non-penalty shots from 2,351 matches, covering the 2015/16 big-five leagues, the last two World Cups and Euros, Copa América 2024, AFCON 2023 and the Messi-era La Liga seasons.

What I did:
🔹 Engineered features from shot geometry, technique and assist type, plus the freeze frame at the moment of the shot (how many defenders are in the shooting lane, where the keeper is)
🔹 Trained a logistic regression and a LightGBM model, splitting train and test by match to avoid leakage
🔹 Benchmarked against StatsBomb's own xG on 471 matches the models never saw

What I found:
✅ LightGBM: log loss 0.2584, AUC 0.809. StatsBomb xG: 0.2577 and 0.811. Statistically indistinguishable on a match-level bootstrap
✅ Better calibrated in aggregate: 1,157 xG for 1,159 actual goals (StatsBomb: 1,084)
✅ Goal angle, distance and defenders in the shooting lane drive the predictions most (SHAP)
✅ Messi scored 100.5 more non-penalty goals than my model expected on 2,335 shots, 6.7 standard deviations above an average finisher. Nobody else in the data comes close.

Notebook, charts and write-up are on GitHub: https://github.com/anthony07-max/football-xg-model

Thanks to StatsBomb for making this data freely available. I'm looking for junior data and sports analytics roles in London, so I'd love to connect with anyone working in football data.

#FootballAnalytics #SportsAnalytics #DataScience #xG #Python #MachineLearning #StatsBomb
