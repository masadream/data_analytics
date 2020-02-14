# ----------------------------------------------------------
# 1.4.1
# ----------------------------------------------------------
# pandas準備
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import ttest_ind


# データセットダウンロード
email_data = pd.read_csv("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

# 女性向けメール配信データをデータセットから除外
male_df = email_data[email_data["segment"] != "Womens E-Mail"]

# 介入を表すtreatment変数を追加
male_df["treatment"] = np.where(male_df["segment"] == "Mens E-Mail", 1, 0)


# ----------------------------------------------------------
# 1.4.2
# ----------------------------------------------------------
# treatmentごとの簡単な集計結果
male_df.groupby("treatment").agg({"conversion" : "mean", "spend" : "mean", "treatment" : "count"})  # カラム名をきちんとなおす

# 元ソースではRでStudentのt検定・両側検定をやっているので、それにしたがう
# https://cran.r-project.org/doc/manuals/r-release/fullrefman.pdf
# https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ttest_ind.html#statsmodels.stats.weightstats.ttest_ind

mens_mail = male_df[male_df["treatment"] == 1]
mens_mail = mens_mail["spend"]
no_mail = male_df[male_df["treatment"] == 0]
no_mail = no_mail["spend"]

rct_ttest = ttest_ind(mens_mail, no_mail, alternative = 'two-sided', usevar = 'pooled')
rct_ttest


# ----------------------------------------------------------
# 1.4.3
# ----------------------------------------------------------

# randomseedの固定
np.random.seed(1)

# 初期値
biased_data = male_df
biased_data["obs_rate_c"] = 0.5
biased_data["obs_rate_t"] = 0.5

# バイアスデータの作成
biased_data["obs_rate_c"] = np.where((biased_data["history"] > 300) | (biased_data["recency"] < 6) | (biased_data["channel"] == "Multichannel"), biased_data["obs_rate_c"], 1)

biased_data["obs_rate_t"] = np.where((biased_data["history"] > 300) | (biased_data["recency"] < 6) | (biased_data["channel"] == "Multichannel"), 1, biased_data["obs_rate_t"])

biased_data["random_number"] = np.random.rand(len(biased_data))

biased_data = biased_data[(biased_data["treatment"] == 0 ) & (biased_data["random_number"] < biased_data["obs_rate_c"]) | (biased_data["treatment"] == 1) & (biased_data["random_number"] < biased_data["obs_rate_t"])]

# バイアスデータのtreatmentごとの簡単な集計
biased_data.groupby("treatment").agg({"conversion" : "mean", "spend" : "mean", "treatment" : "count"})

# バイアスデータでのt検定実施
mens_mail_biased = biased_data[biased_data["treatment"] == 1]
mens_mail_biased = mens_mail_biased["spend"]
no_mail_biased = biased_data[biased_data["treatment"] == 0]
no_mail_biased = no_mail_biased["spend"]

rct_ttest_biased = ttest_ind(mens_mail, no_mail, alternative = 'two-sided', usevar = 'pooled')
rct_ttest_biased
