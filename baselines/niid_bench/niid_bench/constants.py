PATHS = {
    'smoking' : 'data/smoking.csv',
    'heart' : 'data/heart.csv',
    'lumpy' : 'data/lumpy_skin.csv',
    'machine' : 'data/predictive_maintenance.csv',
    'femnist' : 'leaf/data/femnist',
    'synthetic' : 'leaf/data/synthetic',
    'insurance' : 'data/insurance.csv'
}

NUM_FEATURES = {
    'femnist' : 785,
    'synthetic' : 20,
    'heart' : 11,
    'lumpy' : 18,
    'machine' : 7,
    'smoking' : 22,
    'insurance' : 6
}

NUM_CLASSES = {
    'femnist' : 62,
    'synthetic' : 10,
    'heart' : 2,
    'lumpy' : 2,
    'machine' : 2,
    'smoking' : 2,
    'insurance' : 1
}

FEATURES = {
    'heart' : [
        'age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','region'
    ], 
    'lumpy' : [
        'x_coord', 'y_coord', 'region', 'country', 'cld', 'dtr', 'frs',
        'pet', 'pre', 'tms', 'tmp', 'tmx', 'vap', 'wet', 'elevation', 'dominant_land_cover',
        'X5_Ct_2010_Da', 'X5_Bf_2010_Da'
    ],
}

CAT_IDX = {
    'heart': [2, 6, 10, 12, 13],
    'lumpy': [2, 3]
}

CAT_DIM = {
    'heart': [4, 3, 3, 4, 4],
    'lumpy': [3, 39]
}