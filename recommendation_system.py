import sys
print("Python executable being used:", sys.executable)

# Import necessary libraries
try:
    from surprise import Dataset, SVD, SVDpp, NMF
    from surprise.model_selection import cross_validate, GridSearchCV
    print("Surprise library imported successfully!")
except ImportError as e:
    print("ImportError:", e)
    sys.exit(1)  # Exit the script if import fails

# Load the MovieLens 100k dataset
print("Loading the MovieLens 100k dataset...")
try:
    data = Dataset.load_builtin('ml-100k')
    print("Data loaded successfully!")
except Exception as e:
    print(f"Failed to load data: {e}")
    sys.exit(1)

# Check the dataset format
print("Checking dataset format...")
try:
    import pandas as pd
    df = pd.DataFrame(data.raw_ratings, columns=['user', 'item', 'rating', 'timestamp'])
    print("Dataset format checked. Here's a preview:")
    print(df.head())
except Exception as e:
    print(f"Failed to check dataset format: {e}")

# Initialize the SVD algorithm
print("Initializing the SVD algorithm...")
try:
    svd = SVD()
    print("SVD algorithm initialized.")
except Exception as e:
    print(f"Failed to initialize SVD: {e}")
    sys.exit(1)

# Evaluate SVD performance using cross-validation
print("Evaluating SVD performance using cross-validation...")
try:
    svd_cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print("Cross-validation completed. Results:")
    print(svd_cv_results)
except Exception as e:
    print(f"Failed to evaluate SVD performance: {e}")

# Define parameter grid for SVD
print("Defining parameter grid for SVD...")
param_grid = {
    'n_factors': [50, 100, 150],
    'n_epochs': [20, 30],
    'lr_all': [0.002, 0.005],
    'reg_all': [0.02, 0.1]
}

# Perform GridSearchCV
print("Performing GridSearchCV to find the best parameters...")
try:
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    print("GridSearchCV completed.")
    print(f"Best RMSE score: {gs.best_score['rmse']}")
    print(f"Best parameters: {gs.best_params['rmse']}")
except Exception as e:
    print(f"Failed to perform GridSearchCV: {e}")

# SVD++
print("Initializing and cross-validating SVD++ algorithm...")
try:
    svdpp = SVDpp()
    svdpp_results = cross_validate(svdpp, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print("SVD++ cross-validation completed.")
    print(svdpp_results)
except Exception as e:
    print(f"Failed to evaluate SVD++: {e}")

# NMF
print("Initializing and cross-validating NMF algorithm...")
try:
    nmf = NMF()
    nmf_results = cross_validate(nmf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print("NMF cross-validation completed.")
    print(nmf_results)
except Exception as e:
    print(f"Failed to evaluate NMF: {e}")

# Example: Custom collaborative filtering function
print("Running custom collaborative filtering function...")
def collaborative_filtering(X, y, learning_rate=0.01, iterations=1000):
    # Initialization of parameters
    m, n = X.shape
    theta = np.random.randn(n)
    
    for i in range(iterations):
        # Hypothesis
        prediction = np.dot(X, theta)
        # Loss function
        cost = (1 / (2 * m)) * np.sum((prediction - y) ** 2)
        # Gradient descent
        gradient = (1 / m) * np.dot(X.T, (prediction - y))
        theta = theta - learning_rate * gradient
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    
    return theta

# Final statement indicating script completion
print("Script completed successfully!")