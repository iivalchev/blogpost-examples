#%%
import sklearn
from sklearn.model_selection import train_test_split
from functools import partial

# change to cuda if you have one
device = "cpu"

# for illustration, we want it to be fast
n_samples = 1000

X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

split_fn = partial(train_test_split, test_size=0.2)

X_train_raw, X_val_raw, y_train_raw, y_val_raw = split_fn(
    X[:n_samples], y[:n_samples]
)

#%%
from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

regressor_args = {
    "device": device,
    "n_estimators": 2
}

reg = TabPFNRegressor(**regressor_args, fit_mode="batched")

def evaluate_regressor():
    reg_eval = clone_model_for_evaluation(reg, regressor_args, TabPFNRegressor)

    # fit will re-create the preprocecssing as it would be used during inference
    reg_eval.fit(X_train_raw, y_train_raw)

    predictions = reg_eval.predict(X_val_raw)

    mse = mean_squared_error(y_val_raw, predictions)
    mae = mean_absolute_error(y_val_raw, predictions)
    r2 = r2_score(y_val_raw, predictions)

    print(
        f"MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}\n"
    )

#%%
from tabpfn.preprocessing import DatasetCollectionWithPreprocessing

datasets_collection:DatasetCollectionWithPreprocessing = reg.get_preprocessed_datasets(
    X_train_raw, y_train_raw, split_fn, max_data_size=500
)

#%%
from torch.utils.data import DataLoader
from tabpfn.utils import meta_dataset_collator

data_loader = DataLoader(
    datasets_collection, batch_size=1, collate_fn=meta_dataset_collator
)

#%%
from torch.optim import AdamW

optimizer = AdamW(reg.model_.parameters(), lr=1e-5)

#%%
from tqdm import tqdm

evaluate_regressor()

do_epochs = 1
for epoch in range(do_epochs):
    for data_batch in tqdm(data_loader):

        optimizer.zero_grad()

        # extract data and config from the batch
        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            normalized_bardist_,
            bardist_,
            batch_x_test_raw,
            batch_y_test_raw,
        ) = data_batch

        # set the criterion (bar dist.) on the regressor
        reg.normalized_bardist_ = normalized_bardist_[0]

        # fit the regressor from the preprocessed tensors
        reg.fit_from_preprocessed(
            X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
        )

        # foreward pass through the regressor
        averaged_pred_logits, _, _ = reg.forward(
            X_tests_preprocessed
        )

        lossfn = bardist_[0]

        # compute the loss
        nll_loss_per_sample = lossfn(averaged_pred_logits, y_test_standardized.to(device))

        # compute mean loss across all test samples in single forward pass
        loss = nll_loss_per_sample.mean()

        print(f" Loss in EPOCH {epoch + 1}: {loss}")

        loss.backward()

        optimizer.step()

evaluate_regressor()
