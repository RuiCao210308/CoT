Rollout error distribution figures

Inputs are per-sample trajectory rollout metrics. All labels are in English to avoid font portability issues.

Figures:
- rollout_ade_ecdf / rollout_fde_ecdf: ECDF curves. Curves closer to the upper-left are better.
- rollout_ade_scatter_boxplot / rollout_fde_scatter_boxplot: each scatter point is one sample; the boxplot shows median, IQR, whiskers, and long-tail outliers.
- rollout_ade_hist: rollout ADE distribution counts for each model.
- improvement_vs_text_hist: per-sample ADE improvement over openemma_text. Values greater than 0 mean the model is better than openemma_text on that sample.
- scenario_rollout_bar: grouped bar chart for rollout ADE and FDE across all/high-curvature/speed-changing/steady-low-curvature scenarios.

scenario_rollout_bar was generated from explicit scenario/group labels or reconstructed target curvature/speed-change scores.
