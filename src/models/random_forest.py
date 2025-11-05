from sklearn.ensemble import RandomForestRegressor

def get():

    return RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

