import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class kNNMTD():
    def __init__(self, n_obs=100, k=3, random_state=1, sigma_factor=0.25): 
        self.n_obs = n_obs
        self._gen_obs = self.n_obs * 10 
        self.k = k
        self.sigma_factor = sigma_factor
        self.random_state = random_state

    def _gaussian_mf(self, x, u_set, L, U):
        sigma = (U - L) * self.sigma_factor
        return np.exp(-((x - u_set) ** 2) / (2 * sigma ** 2))

    def diffusion(self, sample):
        if isinstance(sample.iloc[0], str):
            sample = sample.astype(float)
        new_sample = []
        n = len(sample)
        min_val = np.min(sample)
        max_val = np.max(sample)
        u_set = (min_val + max_val) / 2

        if u_set == min_val or u_set == max_val:
            Nl = len(sample[sample <= u_set])
            Nu = len(sample[sample >= u_set])
        else:
            Nl = len(sample[sample < u_set])
            Nu = len(sample[sample > u_set])

        skew_l = Nl / (Nl + Nu) if (Nl + Nu) != 0 else 0
        skew_u = Nu / (Nl + Nu) if (Nl + Nu) != 0 else 0
        var = np.var(sample, ddof=1)
        
        if var == 0:
            a = min_val / 5
            b = max_val * 5
            new_sample = np.random.uniform(a, b, size=self._gen_obs)
        else:
            h = var / n
            a = u_set - (skew_l * np.sqrt(-2 * (var/Nl) * np.log(10**(-20)))) if Nl != 0 else min_val
            b = u_set + (skew_u * np.sqrt(-2 * (var/Nu) * np.log(10**(-20)))) if Nu != 0 else max_val
            L = a if a <= min_val else min_val
            U = b if b >= max_val else max_val
            
            while len(new_sample) < self._gen_obs:
                x = np.random.uniform(L, U)
                if x < L or x > U:
                    MF = 0
                else:
                    MF = self._gaussian_mf(x, u_set, L, U)
                rs = np.random.uniform(0, 1)
                if MF > rs:
                    new_sample.append(x)
        return np.array(new_sample)

    def get_global_neighbors(self, val_to_test, X_train, y_train):
        knn = KNeighborsRegressor(n_neighbors=self.k, metric='euclidean')
        knn.fit(X_train, y_train)
        _, nn_indices = knn.kneighbors(val_to_test.reshape(1, -1))
        neighbor_samples = X_train[nn_indices[0]]  
        neighbor_labels = y_train[nn_indices[0]]  
        return neighbor_samples, neighbor_labels

    def fit(self, train, class_col=None):
        train = train.copy()
        train = train.reset_index(drop=True)
        X_train = train.drop(class_col, axis=1).values  
        y_train = train[class_col].values  
        n_features = X_train.shape[1]
        n_samples = len(train)

        surrogate_data = []

        for idx in range(n_samples):
            current_sample = X_train[idx].reshape(1, -1)  
            current_label = y_train[idx]

            neighbor_samples, neighbor_labels = self.get_global_neighbors(current_sample, X_train, y_train)
    
            diffused_features_list = []
            for feat_idx in range(n_features):
                neighbor_feat = neighbor_samples[:, feat_idx]  
                feat_series = pd.Series(neighbor_feat)
                diffused_feat = self.diffusion(feat_series)  
                diffused_features_list.append(diffused_feat)

            diffused_features = np.column_stack(diffused_features_list)

            per_neighbor_samples = self._gen_obs // self.k
            remainder = self._gen_obs % self.k
            labels = []
            for i in range(self.k):

                cnt = per_neighbor_samples + (1 if i < remainder else 0)
                labels.extend([neighbor_labels[i]] * cnt)
            diffused_labels = np.array(labels)  

            assert diffused_features.shape[0] == len(diffused_labels), \
                f"{diffused_features.shape[0]} vs {len(diffused_labels)}"

            temp_df = pd.DataFrame(
                np.hstack([diffused_features, diffused_labels.reshape(-1, 1)]),
                columns=train.columns
            )
            surrogate_data.append(temp_df)

        surrogate_data = pd.concat(surrogate_data, ignore_index=True)
        synth_data = self.sample(surrogate_data, train, class_col)
        return synth_data

    def sample(self, data, real, class_col):

        surrogate_data = data.copy()
        train = real.copy()
        
        for col in train.columns:
            surrogate_data[col] = surrogate_data[col].astype(train[col].dtype, errors='ignore')
        
        surrogate_data = surrogate_data.dropna()
        
        if class_col is not None and len(surrogate_data) > 0:
            unique_classes = np.unique(surrogate_data[class_col])
            div = len(unique_classes)
            if div == 0:
                synth_data = surrogate_data.sample(self.n_obs, replace=True) if not surrogate_data.empty else pd.DataFrame()
            else:
                num = np.abs(self.n_obs)
                class_num = [num // div + (1 if x < num % div else 0) for x in range(div)]
                synth_data = []
                for cls, cnt in zip(unique_classes, class_num):
                    cls_data = surrogate_data[surrogate_data[class_col] == cls]
                    if cls_data.empty:
                        continue
                    try:
                        temp = cls_data.sample(cnt, replace=False)
                    except:
                        temp = cls_data.sample(cnt, replace=True)
                    synth_data.append(temp)
                synth_data = pd.concat(synth_data, ignore_index=True)
        else:
            synth_data = surrogate_data.sample(self.n_obs, replace=True) if not surrogate_data.empty else pd.DataFrame()
        
        for col in train.columns:
            if col in synth_data.columns:
                synth_data[col] = synth_data[col].astype(train[col].dtype, errors='ignore')
        
        return synth_data.dropna().reset_index(drop=True)