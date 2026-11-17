# 日本全国の住居地点における二酸化窒素と夜間道路交通騒音（2015〜2023年度）

論文「Diverging trends in nitrogen dioxide and nighttime road traffic noise in Japanese cities」（松下大輔、ORCID 0000-0002-6765-9366、投稿中）のコードと推計値です。公開データだけを使い、日本の全人口をカバーする約1,230万の住居地点で、2015〜2023年度の各年度の二酸化窒素（NO2）年平均濃度を推計します。同じ地点に姉妹モデル [japan-night-road-noise](https://github.com/m138/japan-night-road-noise) の夜間騒音（Lnight）を付けているので、二つの暴露を同じ住まいで比べられます。

図1〜4は figures/ にあります（src/make_figures.py で作成）。市区町村別のインタラクティブ地図：https://m138.github.io/japan-traffic-no2/ （年度別のNO2、変化率、道路の寄与分、夜間騒音、二重高暴露）。

> 状態：論文は審査中です。推計値：https://doi.org/10.5281/zenodo.24299015 。コード（v1.0.0）：https://doi.org/10.5281/zenodo.24299223 。

## 推計値の入手
推計値はZenodo（https://doi.org/10.5281/zenodo.24299015）でCC BY 4.0により公開しています。主なファイルは次のとおりです。列の定義は data_dictionary/ にあります。
- small_area_no2_noise_2015_2023.csv：2020年国勢調査の小地域（KEY_CODE）ごと
- no2_2015_2023_by_municipality.csv、no2_2015_2023_by_prefecture.csv：市区町村・都道府県ごと
- receptors_no2_noise_2015_2023.parquet：住居地点（1,230万点、経緯度）ごと
- stations_lopo_2015_2023.csv：監視局ごとの実測値と交差検証の予測値
- section_nox_2015_2021.csv：交通センサス区間ごとのNOx排出量

濃度の単位はppb（1 ppb = 1.88 µg/m³）。road_YYYY は1 km以内のセンサス道路で説明される分です。

## 精度
都道府県を一つずつ抜く交差検証（各年度約1,600局）で、全局R² 0.78〜0.81、自排局0.46〜0.60、RMSE約2 ppb。推計は地域や年度の比較のためのもので、個々の住所の濃度の予測には向きません。

## 手順・出典・ライセンス
- スクリプトの順序と引数：PIPELINE.md
- 入力データの出典と利用条件：DATA_SOURCES.md
- コードはMITライセンス、推計値はCC BY 4.0。
