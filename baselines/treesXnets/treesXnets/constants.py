TASKS = {
    'reg_cat_house_sales' : 'regression',
    'maharshipandya/spotify-tracks-dataset' : 'multi',
    'yuweiyin/FinBench' : 'binary',
    'imodels/compas-recidivism' : 'binary',
    'FredZhang7/malicious-website-features-2.4M': 'multi',
    'imodels/diabetes-readmission': 'binary',
}

TARGET = {
    'reg_cat_house_sales' : 'price',
    'maharshipandya/spotify-tracks-dataset' : 'target',
    'yuweiyin/FinBench' : 'target',
    'imodels/compas-recidivism' : 'is_recid',
    'FredZhang7/malicious-website-features-2.4M' : 'is_malicious',
    'imodels/diabetes-readmission': 'readmitted',
}

NUM_CLASSES = {
    'reg_cat_house_sales' : 1,
    'maharshipandya/spotify-tracks-dataset' : 2,
    'yuweiyin/FinBench' : 2,
    'imodels/compas-recidivism' : 2,
    'FredZhang7/malicious-website-features-2.4M' : 3,
    'imodels/diabetes-readmission': 2,
}