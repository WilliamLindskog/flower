TASKS = {
    'reg_cat_house_sales' : 'regression',
    'reg_num_abalone' : 'regression',
    'clf_num_credit' : 'binary',
    'clf_cat_covertype' : 'binary',
}

TARGET = {
    'reg_cat_house_sales' : 'price',
    'reg_num_abalone' : 'Classnumberofrings',
    'clf_num_credit' : 'SeriousDlqin2yrs',
    'clf_cat_covertype' : 'class',
}

NUM_CLASSES = {
    'reg_cat_house_sales' : 1,
    'reg_num_abalone' : 1,
    'clf_num_credit' : 2,
    'clf_cat_covertype' : 2,
}