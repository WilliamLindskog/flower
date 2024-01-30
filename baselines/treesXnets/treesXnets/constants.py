CONSTANTS = {
    'reg_num_abalone' : {
        'task' : 'regression', 
        'target' : 'Classnumberofrings',
        'num_classes' : 1,
    },
    'reg_num_Ailerons' : {
        'task' : 'regression',
        'target' : 'goal',
        'num_classes' : 1,
    },
    'reg_num_cpu_act' : {
        'task' : 'regression',
        'target' : 'usr',
        'num_classes' : 1,
    },
    'reg_num_diamonds' : {
        'task' : 'regression',
        'target' : 'price',
        'num_classes' : 1,
    },
    'reg_num_elevators' : {
        'task' : 'regression',
        'target' : 'Goal',
        'num_classes' : 1,
    },
    'reg_cat_SGEMM_GPU_kernel_performance' : {
        'task' : 'regression',
        'target' : 'Run1',
        'num_classes' : 1,
    },
    'reg_num_houses' : {
        'task' : 'regression',
        'target' : 'medianhousevalue',
        'num_classes' : 1,
    },
    'reg_num_house_16H' : {
        'task' : 'regression',
        'target' : 'price',
        'num_classes' : 1,
    },
    'reg_num_MiamiHousing2016' : {
        'task' : 'regression',
        'target' : 'SALEPRC',
        'num_classes' : 1,
    },
    'reg_num_nyc-taxi-green-dec-2016' : {
        'task' : 'regression',
        'target' : 'tipamount',
        'num_classes' : 1,
    },
    'reg_num_pol' : {
        'task' : 'regression',
        'target' : 'foo',
        'num_classes' : 1,
    },
    'reg_cat_visualizing_soil' : {
        'task' : 'regression',
        'target' : 'track', 
        'num_classes' : 1,
    },
    'reg_num_sulfur' : {
        'task' : 'regression',
        'target' : 'y1',
        'num_classes' : 1,
    },
    'reg_cat_analcatdata_supreme' : {
        'task' : 'regression',
        'target' : 'Log_exposure',
        'num_classes' : 1,
    },
    'reg_cat_topo_2_1' : {
        'task' : 'regression',
        'target' : 'oz267',
        'num_classes' : 1,
    },
    'reg_cat_particulate-matter-ukair-2017' : {
        'task' : 'regression',
        'target' : 'PM.sub.10..sub..particulate.matter..Hourly.measured.',
        'num_classes' : 1,
    },
    'reg_num_wine_quality' : {
        'task' : 'regression',
        'target' : 'quality',
        'num_classes' : 1,
    },
    'reg_num_yprop_4_1' : {
        'task' : 'regression',
        'target' : 'oz252',
        'num_classes' : 1,
    },
    'clf_num_credit' : {
        'task' : 'binary',
        'target' : 'SeriousDlqin2yrs',
        'num_classes' : 2,
    },
    'clf_num_california' : {
        'task' : 'binary',
        'target' : 'price_above_median',
        'num_classes' : 2,
    },
    'clf_num_default-of-credit-card-clients' : {
        'task' : 'binary',
        'target' : 'y',
        'num_classes' : 2,
    },
    'clf_num_Diabetes130US' : {
        'task' : 'binary',
        'target' : 'readmitted',
        'num_classes' : 2,
    },
    'clf_num_electricity' : {
        'task' : 'binary',
        'target' : 'class',
        'num_classes' : 2,
    },
    'clf_num_eye_movements' : {
        'task' : 'binary',
        'target' : 'label',
        'num_classes' : 2,
    },
    'clf_num_heloct' : {
        'task' : 'binary',
        'target' : 'RiskPerformance',
        'num_classes' : 2,
    },
    'clf_num_Higgs' : {
        'task' : 'binary',
        'target' : 'target',
        'num_classes' : 2,
    },
    'clf_num_jannis' : {
        'task' : 'binary',
        'target' : 'class',
        'num_classes' : 2,
    },
    'clf_num_bank-marketing' : {
        'task' : 'binary',
        'target' : 'Class',
        'num_classes' : 2,
    },
    'clf_num_MagicTelescope' : {
        'task' : 'binary',
        'target' : 'class',
        'num_classes' : 2,
    },
    'clf_num_MiniBooNE' : {
        'task' : 'binary',
        'target' : 'signal',
        'num_classes' : 2,
    }
}

# Save the following as a dictionary json in /data
#import json
#with open('./treesXnets/data/constants.json', 'w') as f:
#    json.dump(CONSTANTS, f, indent=4)
#f.close()

#quit()
TASKS = {
    'reg_num_abalone' : 'regression',
    'reg_num_Ailerons' : 'regression',
    'reg_num_cpu_act' : 'regression',
    'reg_num_diamonds' : 'regression',
    'reg_num_elevators' : 'regression',
    'reg_cat_SGEMM_GPU_kernel_performance' : 'regression',
    'reg_num_houses' : 'regression',
    'reg_num_house_16H' : 'regression',
    'reg_num_MiamiHousing2016' : 'regression',
    'reg_num_nyc-taxi-green-dec-2016' : 'regression',
    'reg_num_pol' : 'regression',
    'reg_cat_visualizing_soil' : 'regression',
    'reg_num_sulfur' : 'regression',
    'reg_cat_analcatdata_supreme' : 'regression',
    'reg_cat_topo_2_1' : 'regression',
    'reg_cat_particulate-matter-ukair-2017' : 'regression',
    'reg_num_wine_quality' : 'regression',
    'reg_num_yprop_4_1' : 'regression',
    'clf_num_credit' : 'binary',
    'clf_num_california' : 'binary',
    'clf_num_default-of-credit-card-clients' : 'binary',
    'clf_num_Diabetes130US' : 'binary',
    'clf_num_electricity' : 'binary',
    'clf_num_eye_movements' : 'binary',
    'clf_num_heloct' : 'binary',
    'clf_num_Higgs' : 'binary',
    'clf_num_jannis' : 'binary',
    'clf_num_bank-marketing' : 'binary',
    'clf_num_MagicTelescope' : 'binary',
    'clf_num_MiniBooNE' : 'binary'
}

TARGET = {
    'reg_num_abalone' : 'Classnumberofrings',
    'reg_num_Ailerons' : 'goal',
    'reg_num_cpu_act' : 'usr',
    'reg_num_diamonds' : 'price',
    'reg_num_elevators' : 'Goal',
    'reg_cat_SGEMM_GPU_kernel_performance' : 'Run1',
    'reg_num_houses' : 'medianhousevalue',
    'reg_num_house_16H' : 'price',
    'reg_num_MiamiHousing2016' : 'SALEPRC',
    'reg_num_nyc-taxi-green-dec-2016' : 'tipamount',
    'reg_num_pol' : 'foo',
    'reg_cat_visualizing_soil' : 'track', 
    'reg_num_sulfur' : 'y1',
    'reg_cat_analcatdata_supreme' : 'Log_exposure',
    'reg_cat_topo_2_1' : 'oz267',
    'reg_cat_particulate-matter-ukair-2017' : 'PM.sub.10..sub..particulate.matter..Hourly.measured.',
    'reg_num_wine_quality' : 'quality',
    'reg_num_yprop_4_1' : 'oz252',
    'clf_num_credit' : 'SeriousDlqin2yrs',
    'clf_num_california' : 'price_above_median',
    'clf_num_default-of-credit-card-clients' : 'y',
    'clf_num_Diabetes130US' : 'readmitted',
    'clf_num_electricity' : 'class',
    'clf_num_eye_movements' : 'label',
    'clf_num_heloct' : 'RiskPerformance',
    'clf_num_Higgs' : 'target',
    'clf_num_jannis' : 'class',
    'clf_num_bank-marketing' : 'Class',
    'clf_num_MagicTelescope' : 'class',
    'clf_num_MiniBooNE' : 'signal'
}

NUM_CLASSES = {
    'reg_num_abalone' : 1,
    'reg_num_Ailerons' : 1,
    'reg_num_cpu_act' : 1,
    'reg_num_diamonds' : 1,
    'reg_num_elevators' : 1,
    'reg_cat_SGEMM_GPU_kernel_performance' : 1,
    'reg_num_houses' : 1,
    'reg_num_house_16H' : 1,
    'reg_num_MiamiHousing2016' : 1,
    'reg_num_nyc-taxi-green-dec-2016' : 1,
    'reg_num_pol' : 1,
    'reg_cat_visualizing_soil' : 1,
    'reg_num_sulfur' : 1,
    'reg_cat_analcatdata_supreme' : 1,
    'reg_cat_topo_2_1' : 1,
    'reg_cat_particulate-matter-ukair-2017' : 1,
    'reg_num_wine_quality' : 1,
    'reg_num_yprop_4_1' : 1,
    'clf_num_credit' : 2,
    'clf_num_california' : 2,
    'clf_num_default-of-credit-card-clients' : 2,
    'clf_num_Diabetes130US' : 2,
    'clf_num_electricity' : 2,
    'clf_num_eye_movements' : 2,
    'clf_num_heloct' : 2,
    'clf_num_Higgs' : 2,
    'clf_num_jannis' : 2,
    'clf_num_bank-marketing' : 2,
    'clf_num_MagicTelescope' : 2,
    'clf_num_MiniBooNE' : 2
}