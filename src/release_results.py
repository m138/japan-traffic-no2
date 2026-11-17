"""Adds the population-weighted arithmetic mean of Lnight to the small-area file and writes the English
summary tables in data/results/ from the analysis outputs (unified_stats.py, measured_trends*.py,
cutoff_sensitivity.py, popalloc_building.py, sens_kriging.py, sens_emission.py, compare_external.py).
Usage: release_results.py ANALYSIS_DIR RELEASE_DATA_DIR
  ANALYSIS_DIR holds results_unified/, rq34/, sensitivity/ and comparison_external/."""
import pandas as pd, numpy as np, os, sys
R, DATA = sys.argv[1].rstrip('/') + '/', sys.argv[2]
os.chdir(DATA)
d=pd.read_parquet('receptors_no2_noise_2015_2023.parquet',columns=['KEY_CODE','pop_w','Lnight_2021'])
d['wL']=d.pop_w*d.Lnight_2021
g=d.groupby('KEY_CODE')[['pop_w','wL']].sum(); am=(g.wL/g.pop_w).round(2)
S=pd.read_csv('small_area_no2_noise_2015_2023.csv',dtype={'KEY_CODE':str})
if 'Lnight_2021_mean' in S: S=S.drop(columns='Lnight_2021_mean')
S.insert(S.columns.get_loc('Lnight_2021_energy_mean'),'Lnight_2021_mean',S.KEY_CODE.map(am))
S.to_csv('small_area_no2_noise_2015_2023.csv',index=False)
# results tables in English
os.makedirs('results',exist_ok=True)
t=pd.read_csv(R+'rq34/measured_trends_2015_2023.csv')
t['series']=t.series.map({'夜間騒音（実測、dB差）':'nighttime_noise_dB','昼間騒音（実測、dB差）':'daytime_noise_dB','NO2 自排局（%変化）':'NO2_roadside_pct','NO2 一般局（%変化）':'NO2_ambient_pct'})
t.rename(columns={'est':'change_from_2015','lo':'ci_low','hi':'ci_high'}).to_csv('results/measured_trends_2015_2023.csv',index=False)
t=pd.read_csv(R+'rq34/robust/measured_trends_robustness.csv'); t.columns=['series','sample','change_2019','change_2023','ci_low_2023','ci_high_2023','sites','records']
t['series']=t.series.map({'夜間騒音（dB差）':'nighttime_noise_dB','NO2 自排局（%）':'NO2_roadside_pct','NO2 一般局（%）':'NO2_ambient_pct'})
t['sample']=t['sample'].map({'全地点（2回以上）':'measured in two or more years (main)','2015と2023の両年度に測定':'measured in 2015 and 2023','5年度以上測定':'measured in five or more years','全局（2回以上）':'measured in two or more years (main)','9年度すべて測定':'measured in all nine years'})
t.to_csv('results/measured_trends_robustness.csv',index=False)
t=pd.read_csv(R+'rq34/robust/sites_by_year.csv'); t.columns=['fiscal_year','noise_sites','noise_sites_also_previous_year','roadside_stations','ambient_stations']; t.to_csv('results/sites_by_year.csv',index=False)
t=pd.read_csv(R+'sensitivity/noise_cutoff/cutoff_500m_comparison.csv'); t.columns=['indicator','cutoff_200m_main','cutoff_500m']
t['indicator']=['Lnight>=55dB_pct','high_coexposure_pct','high_coexposure_10k_people','of_which_large_cities_pct','of_which_within_50m_pct','of_which_within_100m_pct','point_correlation','municipal_mean_correlation','within_municipality_correlation','noise_variance_between_municipalities_pct','NO2_variance_between_municipalities_pct','covariance_between_municipalities_pct','high_coexposure_by_small_area_means_pct','point_high_coexposure_missed_by_small_area_means_pct','small_area_means_only_pct_of_population']
t.to_csv('results/sensitivity_cutoff_500m.csv',index=False)
t=pd.read_csv(R+'sensitivity/noise_cutoff/monitoring_sites_200_vs_500.csv'); t.columns=['cutoff','records','bias_dB','sd_dB','r','rmse_dB','share_within_3dB']; t['cutoff']=['200 m','500 m']; t.to_csv('results/noise_validation_cutoffs.csv',index=False)
t=pd.read_csv(R+'sensitivity/popalloc/popalloc_building_comparison_national.csv',dtype={'都府県':str})
t.columns=['pref','placement','pop_10k','within50m_pct','within100m_pct','Lnight_mean','Lnight55_pct','NO2_mean_ppb','NO2_20ug_pct','high_coexposure_pct','of_which_within50m_pct','of_which_within100m_pct','of_which_large_city_wards_pct','r_noise_NO2']
t['placement']=t.placement.map({'元の配置（小地域内に一様）':'uniform (main)','建物用地に配分':'building land'}); t['pref']=t.pref.replace({'全国計':'Japan'})
t.to_csv('results/sensitivity_building_land_placement.csv',index=False)
k=pd.read_csv(R+'sensitivity/sens_kriging.csv'); k['spec']=k.spec.str.split(' ').str[0]; k.to_csv('results/sensitivity_kriging.csv',index=False)
e=pd.read_csv(R+'sensitivity/sens_emission_cv.csv'); e['case']=e.case.str.split(' ').str[0]; tot=pd.read_csv(R+'sensitivity/sens_emission_totals.csv'); tot['case']=tot.case.str.split(' ').str[0]
e.merge(tot,on='case').to_csv('results/sensitivity_emissions.csv',index=False)
c=pd.read_csv(R+'comparison_external/compare_external_stations.csv'); c['method']=c.method.str.replace('（年ずれ）','_yearlag'); c.to_csv('results/comparison_external_models.csv',index=False)

U=R+'results_unified/'
c=pd.read_csv(U+'correlation_by_level.csv'); c['単位']=['residential points','small areas','municipalities','secondary medical care areas','prefectures','residential points (deviations within municipalities)']; c.columns=['level','n','r']; c.to_csv('results/correlation_by_level.csv',index=False)
t=pd.read_csv(U+'table1_by_municipality_type.csv'); t.columns=['municipality_type','pop_million','NO2_2022_ppb','NO2_change_2015_2023_pct','Lnight55_pct','NO2_20ug_pct','high_coexposure_pct','share_of_all_high_coexposure_pct','r_noise_NO2']
t['municipality_type']=["Tokyo's 23 wards",'designated cities','cities >=300,000','cities 100,000-300,000','cities 50,000-100,000','cities <50,000','towns and villages','Japan']; t.to_csv('results/table1_by_municipality_type.csv',index=False)
k=pd.read_csv(U+'key_numbers.csv'); k.columns=['indicator','value']
en=['high_coexposure_pct','high_coexposure_10k_people','Lnight55_pct','NO2_20ug_pct','NO2_20ug_among_Lnight55_pct','NO2_2022_ppb','above_WHO10ug_2022_pct','above_WHO10ug_2015_pct','above_WHO10ug_2023_pct','NO2_2015_ppb','NO2_2023_ppb','near_road_change_2015_2023_pct','of_high_coexposure_Tokyo23_pct','of_high_coexposure_designated_cities_pct','of_high_coexposure_within50m_pct','of_high_coexposure_within100m_pct','nearest_road_prefectural_pct','nearest_road_national_highway_pct','nearest_road_motorway_pct','nearest_road_municipal_other_pct','high_coexposure_dist_lt20m_pct','high_coexposure_dist_20_50m_pct','high_coexposure_dist_50_100m_pct','high_coexposure_dist_100_200m_pct','high_coexposure_dist_ge200m_pct','high_coexposure_by_small_area_means_pct','noise_variance_between_municipalities_pct','NO2_variance_between_municipalities_pct','covariance_between_municipalities_pct','within_municipality_r','between_municipality_r','r_noise_near_road_NO2','r_noise_background_NO2']
assert len(en)==len(k); k['indicator']=en; k.to_csv('results/key_numbers.csv',index=False)
print(sorted(os.listdir('results')))
