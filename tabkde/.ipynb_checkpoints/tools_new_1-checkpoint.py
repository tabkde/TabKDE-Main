from sklearn.preprocessing import StandardScaler
# from scipy.spatial import KDTree

# from sklearn.metrics import mutual_info_score
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np
# from sklearn.decomposition import PCA
# import os
# import json

# def load_files(data_path):
#     # File paths
#     train_file_path = data_path+'real.csv'
#     test_file_path = data_path+'test.csv'
#     synth_tabsyn_path = data_path+'tabsyn.csv'

#     # Load CSV files into DataFrames
#     try:
#         train_df = pd.read_csv(train_file_path)
#         test_df = pd.read_csv(test_file_path)
#         synth_tabsyn = pd.read_csv(synth_tabsyn_path)
#         return train_df, test_df, synth_tabsyn
#     except Exception as e:
#         print(f"Error loading files: {e}")
#         return None, None

class DataProcessor:
    def __init__(self):
        self.scaler = None
        self.encoding_map = {}
        self.target_column = None
        self.decorrelation = None
        self.U = None
        self.eigens = None
        self.json_path = None 
        self.columns = None
        self.cat_columns = None
        self.num_columns = None

    def fit(self, df_train, target_column = None, normalize=True, decorrelation = False, json_path = None):
        self.decorrelation = decorrelation
        df_train_encoded = df_train.copy()
        self.json_path = json_path
        data = {}
     
        # Check if JSON file exists
        if json_path is not None and os.path.exists(self.json_path):
            # Load the JSON file
            with open(self.json_path, 'r') as file:
                data = json.load(file)

            # Extract categorical column indices and names
            if "cat_col_idx" in data and "column_names" in data:
                self.columns = data["column_names"]
                self.cat_columns = [data["column_names"][idx] for idx in data["cat_col_idx"]]
            
            # Extract num column indices and names
            if "num_col_idx" in data and "column_names" in data:
                self.num_columns = [data["column_names"][idx] for idx in data["num_col_idx"]]
            
            if data["task_type"] != "regression":
                self.cat_columns.extend([data["column_names"][idx] for idx in data["target_col_idx"]])
            elif data["task_type"] == "regression":
                self.num_columns.extend([data["column_names"][idx] for idx in data["target_col_idx"]])
            else: 
                ttt = data["task_type"]
                raise ValueError(f"task_type = {ttt} is unknown")
                
        else:
            self.columns = df_train.columns
            self.cat_columns = df_train.select_dtypes(include=['object']).columns
            self.num_columns = df_train.select_dtypes(include=['number']).columns

         
        for col in self.cat_columns:
            self.encoding_map[col] = {cat: code for code, cat in enumerate(df_train[col].astype('category').cat.categories, start=1)}
            df_train_encoded[col] = df_train[col].map(self.encoding_map[col])
            
        if self.target_column is not None:
            train_target = df_train_encoded.pop(target_column).to_numpy().astype(int)
        else: 
            train_target = None
            
        df_train_encoded = df_train_encoded.astype(float)

        if normalize:
            self.scaler = StandardScaler()
            df_train_encoded[:] = self.scaler.fit_transform(df_train_encoded)

        if self.decorrelation:
            pca = PCA(n_components = df_train_encoded.shape[1])
            pca.fit(df_train_encoded)
            self.U = pca.components_
            self.eigens = pca.explained_variance_
            df_train_encoded[:] = df_train_encoded@self.U.T
            
        return df_train_encoded, train_target
        
    def transform(self, df_test):
        df_test_encoded = df_test.copy()
        for col in self.cat_columns:
            df_test_encoded[col] = df_test[col].map(self.encoding_map[col])
            
        if self.target_column is not None:
            test_target = df_test_encoded.pop(target_column).to_numpy().astype(int)
        else: 
            test_target = None
            
        df_test_encoded = df_test_encoded.astype(float)

        if self.scaler is not None:
            df_test_encoded[:] = self.scaler.transform(df_test_encoded)
        if self.decorrelation:
            df_test_encoded[:] = df_test_encoded@self.U.T
         
        return df_test_encoded, test_target

    def decode(self, num_data):
        """
        Reverts the scaling and encoding back to the original data settings.
        """
     
        df_decoded = num_data.copy()
        if self.decorrelation:
            print('decorrelation is ON')
            df_decoded[:] = df_decoded@self.U
        
        if self.scaler is not None:
            df_decoded[:] = self.scaler.inverse_transform(df_decoded)

        for col, mapping in self.encoding_map.items():
            reverse_mapping = {code: cat for cat, code in mapping.items()}
            min_key = min(reverse_mapping.keys())
            max_key = max(reverse_mapping.keys())
            df_decoded[col] = df_decoded[col].round().astype(int).apply(
                lambda x: reverse_mapping.get(
                    min(max(x, min_key), max_key), reverse_mapping[min_key]
                )
            )

        return df_decoded
    def get_columns(self):
         """
         return columns, cat columns, and num columns 
         """
         return self.columns, self.cat_columns, self.num_columns



class EmpiricalTransformer:
    def __init__(self, df):
        """
        Initialize the EmpiricalTransformer with a DataFrame.
        
        Parameters:
        - df: A pandas DataFrame with k columns.
        """
        self.df = df
        self.df_sorted = None
        self.df_ranks = None
        self.fit()
    
    def fit(self, method = 'min'):
        """
        method: it determines who to treat with ties in ranking (min, max, average, dense)
        Fit the model on the given DataFrame:
        1. Sort each column and create df_sorted.
        2. Generate a dataframe of ranks for each entry in the original df.
        """
        self.df_sorted = self.df.apply(np.sort, axis=0)
        self.df_ranks = self.df.rank(method= method).astype(int)/self.df.shape[0] 
        #method = 'dense' : Assigns the minimum rank to all tied values and ranks are consecutive (no gaps between ranks)
        return self.df_ranks
    
    def inverse_empirical(self, u, sorted_col):
        """
        Transform a uniform random value using the inverse empirical distribution of a column.
        
        Parameters:
        - u: A uniform random value in [0, 1].
        - sorted_col: A sorted numpy array or pandas Series.
        
        Returns:
        - A value sampled from the empirical distribution.
        """
        n = len(sorted_col)
        ecdf = np.arange(1, n + 1) / n  # ECDF values
        return np.interp(u, ecdf, sorted_col)
    
    def convert(self, u_vectors):
        """
        Transform random vectors using the inverse empirical distribution of each column.
        
        Parameters:
        - u_vectors: A 2D numpy array where each row is a random vector (u_1, ..., u_k).
        
        Returns:
        - transformed_vectors_df: A DataFrame of transformed vectors with the same column names as df.
        """
        transformed_vectors = []
        for u_vec in u_vectors:
            transformed_vec = [
                self.inverse_empirical(u, self.df_sorted.iloc[:, i].values)
                for i, u in enumerate(u_vec)
            ]
            transformed_vectors.append(transformed_vec)
        
        transformed_vectors = np.array(transformed_vectors)
        transformed_vectors_df = pd.DataFrame(transformed_vectors, columns=self.df.columns)
        return transformed_vectors_df

def perturb_vector_with_partial_resampling_with_DCP(X, model, sigma = .2, L_limits=None, U_limits=None, l = 2):
    """
    Perturb a given vector x within a d-dimensional hypercube [0, 1]^d.

    This function modifies the input vector x by:
    1. Sampling a random vector u = (u_1, ..., u_d) from a normal distribution.
    2. Adding u to x to compute x'.
    3. Resampling only the out-of-bound coordinates of x' until all coordinates are within [0,1].

    Parameters:
    - x: np.ndarray of shape (n, d), the input vectors to be perturbed.

    Returns:
    - x_prime: np.ndarray of shape (n, d), the perturbed vectors within [0,1]^d.
    """
    n,d = X.shape
    if L_limits is None:
        L_limits = np.zeros(d)
    if U_limits is None:
        U_limits = np.ones(d)
    if isinstance(sigma, (int, float, np.float64, np.int32, np.int64)):  # Check if sigma is a real value
        Sigma = sigma * np.identity(d)
    elif isinstance(sigma, np.ndarray):  # Check if sigma is already a matrix
        Sigma = sigma
    elif sigma == 'local':
        tree = KDTree(X)
    elif sigma == 'global':
        Sigma = np.cov(X.T)
    
    else:
        raise TypeError("Sigma must be a real number or a numpy matrix/array.")
        
    synth = []
    counter = 0
    while counter < len(X):
        x = X[counter]
        if isinstance(sigma, str) and sigma == 'local':
            _, I= tree.query(x, k = l*d)
            NN = X[I].T
            Sigma = np.cov(NN)
        x_prime = gen_one_point_at(x, model, Sigma,  L_limits, U_limits)
        if x_prime is not None:
            synth.append(x_prime)
            counter+= 1
            # if counter %2000 == 0:
            #     print(counter)
    return np.array(synth)
    
    
def gen_one_point_at(x, model, Sigma,  L_limits, U_limits):
    d = x.shape[0]
    # print(d)
    x_prime = np.zeros(x.shape)
    counter = 0
    outside_bounds = np.ones(x_prime.shape, dtype=bool)
    r = -1
    while r < 0:
        r = model.sample(1)[0][0][0]
    normilizer = 1
    flag = False
    while counter < 10 * d:
        d_new = sum(outside_bounds)
        # print(d_new)
        if flag:
            return None 
        if d_new == 1:
            flag = True
            u *= -1
        else:
            u = np.random.multivariate_normal(np.zeros(d), Sigma)
            u /= np.linalg.norm(u)
        x_prime[outside_bounds] = x[outside_bounds] + normilizer*r*u[outside_bounds]
        outside_bounds = (x_prime < L_limits) | (x_prime > U_limits)
        # print(outside_bounds)
        normilizer = np.linalg.norm(u[outside_bounds])
        if not np.any(outside_bounds):
            return x_prime
    return None  


# from sklearn.metrics import mutual_info_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# def compute_correlation_divergence(real_data, synthetic_data, show_numbers=True, title = None, vmin = 0, vmax = 1):
#     """
#     Compute the absolute divergence heatmap between real and synthetic data correlations.

#     Parameters:
#     real_data (pd.DataFrame): Real dataset with mixed features.
#     synthetic_data (pd.DataFrame): Synthetic dataset with mixed features.
#     show_numbers (bool): Whether to display numbers in heatmap squares.

#     Returns:
#     None: Displays a heatmap of the divergence.
#     """
#     def mixed_correlation(df):
#         """Compute pairwise correlations for mixed-type data."""
#         columns = df.columns
#         n = len(columns)
#         corr_matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)

#         for i, col1 in enumerate(columns):
#             for j, col2 in enumerate(columns):
#                 if i <= j:  # Compute only once for each pair
#                     if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
#                         # Real-to-Real: Pearson correlation
#                         corr = df[col1].corr(df[col2])
#                     elif isinstance(df[col1].dtype, pd.CategoricalDtype) and isinstance(df[col2].dtype, pd.CategoricalDtype):
#                         # Categorical-to-Categorical: Mutual Information
#                         corr = mutual_info_score(df[col1].astype(str), df[col2].astype(str))
#                     else:
#                         # Real-to-Categorical: Bin the numerical column and compute Mutual Information
#                         if pd.api.types.is_numeric_dtype(df[col1]):
#                             try:
#                                 binned = pd.cut(df[col1].astype(float), bins=10, labels=False)
#                                 corr = mutual_info_score(binned, df[col2].astype(str))
#                             except ValueError:
#                                 corr = 0  # Handle empty or incompatible data
#                         else:
#                             try:
#                                 binned = pd.cut(df[col2].astype(float), bins=10, labels=False)
#                                 corr = mutual_info_score(binned, df[col1].astype(str))
#                             except ValueError:
#                                 corr = 0  # Handle empty or incompatible data
#                     corr_matrix.iloc[i, j] = corr
#                     corr_matrix.iloc[j, i] = corr
#         return corr_matrix

#     # Ensure all categorical columns in synthetic and real data are categorical
#     def ensure_categorical(df, reference_df, vmin = 0, vmax = 1):
#         for col in reference_df.select_dtypes(include=['category']).columns:
#             df[col] = df[col].astype('category')

#     ensure_categorical(synthetic_data, real_data)

#     # Compute correlation matrices for real and synthetic data
#     real_corr = mixed_correlation(real_data)
#     synthetic_corr = mixed_correlation(synthetic_data)

#     # Compute absolute divergence
#     divergence = np.abs(real_corr - synthetic_corr)

#     # Plot heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(divergence, annot=show_numbers, cmap="Greens", cbar=True, 
#                 fmt=".2f" if show_numbers else None, linewidths=0.5, linecolor='white', 
#                 vmin=vmin,  # Set minimum value for the colour scale
#                 vmax=vmax   # Set maximum value for the colour scale
#                )
#     plt.title(f"Absolute Divergence in Pairwise Correlations\n({title})")
#     plt.show()

# #----------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# from scipy.stats import chi2_contingency
# import seaborn as sns
# import matplotlib.pyplot as plt


# import numpy as np
# import pandas as pd
# from scipy.stats import ks_2samp, pearsonr
# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import KBinsDiscretizer
# from scipy.spatial.distance import jensenshannon


# import argparse
# import sys
# import os
# # minlength = 100

# def kolmogorov_smirnov_test(real_col, synthetic_col):
#     return ks_2samp(real_col, synthetic_col).statistic

# def total_variation_distance(real_col, synthetic_col):
#     minlength = max(len(np.unique(real_col)), len(np.unique(synthetic_col)))
#     real_dist = np.bincount(real_col, minlength = minlength) / len(real_col)
#     synthetic_dist = np.bincount(synthetic_col, minlength = minlength) / len(synthetic_col)
#     return np.sum(np.abs(real_dist - synthetic_dist)) / 2

# def pearson_correlation(real_col1, real_col2, synthetic_col1, synthetic_col2):
#     real_corr = pearsonr(real_col1, real_col2)[0]
#     synthetic_corr = pearsonr(synthetic_col1, synthetic_col2)[0]
#     return abs(real_corr - synthetic_corr)


# def contingency_similarity(real_cat1, real_cat2, synthetic_cat1, synthetic_cat2):
#     real_contingency = pd.crosstab(real_cat1, real_cat2)
#     synthetic_contingency = pd.crosstab(synthetic_cat1, synthetic_cat2)
    
#     # Align shapes by reindexing
#     real_contingency = real_contingency.reindex(index=synthetic_contingency.index, columns=synthetic_contingency.columns, fill_value=0)
#     synthetic_contingency = synthetic_contingency.reindex(index=real_contingency.index, columns=real_contingency.columns, fill_value=0)
    
#     return jensenshannon(real_contingency.values.flatten(), synthetic_contingency.values.flatten())


# def bucketize_and_contingency(real_num, real_cat, synthetic_num, synthetic_cat, n_bins=10):
#     discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
#     real_num_binned = discretizer.fit_transform(np.array(real_num).reshape(-1, 1)).astype(int).flatten()
#     synthetic_num_binned = discretizer.fit_transform(np.array(synthetic_num).reshape(-1, 1)).astype(int).flatten()
#     return contingency_similarity(real_num_binned, real_cat, synthetic_num_binned, synthetic_cat)

# def evaluate_column_wise(real_data, synthetic_data, Columns, num_columns):
#     results = {}
#     for col in Columns:
#         if col in num_columns:
#             results[col] = kolmogorov_smirnov_test(real_data[col], synthetic_data[col])
#         else:
#             results[col] = total_variation_distance(real_data[col].astype('category').cat.codes,
#                                                     synthetic_data[col].astype('category').cat.codes)
#     return results

# def evaluate_pair_wise(real_data, synthetic_data, columns, num_columns, cat_columns):
#     results = {}
#     for col1 in columns:
#         for col2 in columns:
#             if col1 == col2:
#                 continue
#             if col1 in num_columns and col2 in num_columns:
#                 # print('num, num', col1, col2)
#                 results[(col1, col2)] = pearson_correlation(real_data[col1], real_data[col2],
#                                                             synthetic_data[col1], synthetic_data[col2])
#             elif col1 in cat_columns and col2 in cat_columns:
#                 # print('cat, cat', col1, col2)
#                 results[(col1, col2)] = contingency_similarity(real_data[col1], real_data[col2],
#                                                                synthetic_data[col1], synthetic_data[col2])
#             else:
#                 # print('cat, num', col1, col2)
#                 num_col = col1 if col1 in num_columns else col2
#                 cat_col = col2 if col2 in cat_columns else col1
#                 results[(col1, col2)] = bucketize_and_contingency(real_data[num_col], real_data[cat_col],
#                                                                   synthetic_data[num_col], synthetic_data[cat_col])
#     return convert_to_dataframe(results, columns, num_columns)


# def convert_to_dataframe(results, columns, num_columns):
#     # Sort columns with numerical columns first
#     sorted_columns = sorted(columns, key=lambda x: (x not in num_columns, x))
    
#     # Create a DataFrame indexed and columned by sorted columns
#     df = pd.DataFrame(index=sorted_columns, columns=sorted_columns)
    
#     # Fill the DataFrame with the results
#     for (col1, col2), value in results.items():
#         df.loc[col1, col2] = value

#     return df

# def plot_column_wise_heatmap(column_wise_results, title = None):
#     columns = list(column_wise_results.keys())
#     values = list(column_wise_results.values())
   
#     df = pd.DataFrame(values, index=columns, columns=['KST/TVD'])
   
#     plt.figure(figsize=(10, len(columns) * 0.5))
#     sns.heatmap(df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Difference'},
#         vmin=0,   # Set minimum colour range
#         vmax=1    # Set maximum colour range
# )
#     plt.title('Column-Wise Results (KST for numerical, TVD for categorical)' if title is None else f'Column-Wise Results (KST for numerical, TVD for categorical)\n {title}' )
#     plt.xlabel('Metric')
#     plt.ylabel('Columns')
#     plt.show()


# def load_json_as_dict(json_file_path):
#     """
#     Loads a JSON file and returns its contents as a dictionary.

#     Parameters:
#         json_file_path (str): The path to the JSON file.

#     Returns:
#         dict: The contents of the JSON file as a dictionary.
#     """
#     try:
#         with open(json_file_path, 'r') as file:
#             data = json.load(file)
#         return data
#     except FileNotFoundError:
#         print(f"Error: The file at {json_file_path} was not found.")
#     except json.JSONDecodeError as e:
#         print(f"Error: Failed to decode JSON. {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# def pretty_print_dict(dictionary):
#    """
#    Pretty prints a dictionary in a readable JSON format.
#    """
#    print(json.dumps(dictionary, indent=4, sort_keys=True))


# def load_data_files(quality_file: str, shape_file: str, trend_file: str, coverage_file: str):
#     """
#     Load data from quality.txt, shape.csv, trend.csv, and coverage.csv.

#     Args:
#         quality_file (str): Path to the quality.txt file.
#         shape_file (str): Path to the shape.csv file.
#         trend_file (str): Path to the trend.csv file.
#         coverage_file (str): Path to the coverage.csv file.

#     Returns:
#         tuple: A tuple containing:
#             - quality_data (list of str): Lines from quality.txt.
#             - shape_data (pd.DataFrame): DataFrame from shape.csv.
#             - trend_data (pd.DataFrame): DataFrame from trend.csv.
#             - coverage_data (pd.DataFrame): DataFrame from coverage.csv.
#     """
#     # Load quality.txt
#     try:
#         with open(quality_file, 'r') as file:
#             quality_data = [line.strip() for line in file.readlines()]
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {quality_file}")

#     # Load shape.csv
#     try:
#         shape_data = pd.read_csv(shape_file)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {shape_file}")

#     # Load trend.csv
#     try:
#         trend_data = pd.read_csv(trend_file)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {trend_file}")

#     # Load coverage.csv
#     try:
#         coverage_data = pd.read_csv(coverage_file)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {coverage_file}")

#     return quality_data, shape_data, trend_data, coverage_data


# def create_dataframe_from_nested_dict(nested_dict):
#     """
#     Creates a DataFrame where:
#     - Columns are indexed by keys of the main dictionary.
#     - Rows are indexed by the keys of the inner 'XGBClassifier' dictionary.

#     Parameters:
#         nested_dict (dict): The input nested dictionary.

#     Returns:
#         pd.DataFrame: The transformed DataFrame.
#     """
#     # Extract column keys (outer dictionary keys)
#     columns = nested_dict.keys()
#     index_name = list(nested_dict[list(columns)[0]].keys())[0]

#     # Extract row keys from 'XGBClassifier' of the first column
#     # (assuming all 'XGBClassifier' keys are the same across columns)
#     first_key = next(iter(columns))
#     row_keys = nested_dict[first_key][index_name].keys()

#     # Create a DataFrame with row keys as the index
#     df = pd.DataFrame(index=row_keys)

#     # Populate the DataFrame with values from the nested dictionary
#     for col in columns:
#         df[col] = pd.Series(nested_dict[col][index_name])
     
#     # Set the name for the row index
#     df.index.name = index_name

#     return df


# def load_npy_file(file_path):
#     try:
#         data = np.load(file_path)
#         print("Shape of the data:", data.shape)
#         print("Data type:", data.dtype)
        
#         return data
#     except Exception as e:
#         print("An error occurred:", e)

