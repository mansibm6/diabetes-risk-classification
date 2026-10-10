# Diabetes model comparison

Prepares a CVD survey dataset and compares logistic regression, random forest, SVM, KNN, and LDA classifiers.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [Data_Preprocessing_Mansib_Final.ipynb](Data_Preprocessing_Mansib_Final.ipynb)
- [KNN_YiyunSheng_Final.ipynb](KNN_YiyunSheng_Final.ipynb)
- [LDA_YiyunSheng_Final.ipynb](LDA_YiyunSheng_Final.ipynb)
- [SVM_Hasti_Final.ipynb](SVM_Hasti_Final.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `CVD.csv`

## Outputs

Scaled train/test CSVs, classification reports, ROC curves, feature rankings, and confusion matrices.

## Implementation and limitations

Run Data_Preprocessing_Mansib_Final.ipynb first. Logistic/RF/SVM consume X_train_scaled_balanced.csv, X_test_scaled.csv, y_train_balanced.csv, and y_test.csv. KNN/LDA are separate experiments requiring Features.csv and Outcome.csv. The RF script now defaults to those CSV outputs; --format npy supports the original numeric arrays. Global categorical encoding and pre-CV SMOTE in the historical workflow limit the validity of cross-validation scores; use fold-local preprocessing for a new scientific benchmark.

CLI example: `python LR_Mohammad_Final.py --data-dir /path/to/preprocessed-files`. RF and tuning scripts accept the same options.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.
