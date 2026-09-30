# TabDiffuse user guide

TabDiffuse is a program on your computer that learns a table and creates new rows with the same kind of columns. It then checks whether those rows look like the original table, whether they copy real records, and whether they are useful for a simple prediction task.

Your file stays on this machine. Nothing is uploaded to an outside service. A privacy result in the app is a measurement, not a promise that the new file is anonymous.

## Open the app

1. Open PowerShell.
2. Go to the project folder and start the app:

```powershell
cd "d:\QuorentX\College Projects\TABDIFFUSE"
.\.venv\Scripts\activate
streamlit run app.py
```

3. Open the address PowerShell prints. It is usually http://localhost:8501.
4. Use the list on the left. Work from the top of that list toward the bottom.

If the page is already open, refresh it after the app starts.

## What you will need

Use your own CSV, TSV, Excel, or Parquet file. Open **Dataset Upload**, choose the file, and click **Analyze dataset**.

## Do the steps in this order

### 1. Home

You can type a research question, a hypothesis, and notes on this page. Those notes are saved with the report. The table itself is loaded from **Dataset Upload**. After that page analyzes the file, click **Continue to profiling**.

### 2. Data Profiling

Read the number of rows, missing values, duplicates, and warnings. This page does not delete anything.

Click **Continue to preprocessing**.

### 3. Preprocessing

Click **Apply preprocessing**.

The app keeps three groups of rows:

- training rows, which teach the model
- validation rows, which pick the best saved model
- holdout rows, which are kept back for the later tests

Click **Continue to model configuration**.

### 4. Model Configuration

Leave the demo settings for a first run, or click **Use recommended configuration**.

Click **Continue to training**.

### 5. Training

Click **Start training**.

The chart shows the real loss after each epoch. **Pause**, **Resume**, and **Stop training** are available while it runs.

When training finishes, the left side shows an experiment id such as `EXP-2026-0001`. That id is the name of this run. A new run gets a new id. It does not replace an older one.

### 6. Synthetic Generation

Choose how many new rows you want. For a first look, a few hundred rows is enough.

Leave **Apply constraints** unchecked the first time. The rules in the box are examples. They run only after you check that box.

Click **Generate synthetic data**.

The new table is saved as:

- `data/synthetic/<experiment id>/synthetic.csv`
- `data/synthetic/<experiment id>/synthetic.parquet`

You can also download those files from the page.

The page tells you if any numbers were moved back into the range of the training table, and if any dates were repaired into a real calendar day.

Click **Run evaluation** when the table is ready.

### 7. Quality Evaluation

Click **Run quality evaluation**.

This compares the new rows with the training rows: numbers, categories, and correlations. Charts are there to look at. They are not a privacy certificate.

### 8. Privacy Evaluation

Click **Run privacy diagnostics**.

You can name quasi-identifiers and one sensitive column first. The page shows:

- how many new rows are exact copies
- how close new rows are to real rows
- whether new rows look more like the training rows than the held-out rows
- whether a sensitive column can be guessed from the new rows

The labels are Low, Moderate, or Higher observed risk. You can change the cutoffs later in Settings. These labels are not differential privacy.

### 9. ML Utility

Pick the target column, such as `loan_status` or `churn_status`.

Click **Run utility evaluation**.

The app trains simple models on real rows, on the new rows, and on both, then tests them on real rows that were held out. It does not announce a winner. Read the table yourself.

### 10. Experiment Comparison

Click **Run baseline comparison**.

This places TabDiffuse next to simpler ways of making rows: copying random training rows, a basic Gaussian sample, and a Gaussian copula. Copying rows can look accurate and still be a poor privacy result, because the exact-match rate shows the copies.

### 11. Reports

Click **Generate HTML and PDF report**.

The files are saved in `data/reports/<experiment id>/`.

## Buttons you may need later

| Button | When to use it |
|---|---|
| Pause / Resume | You want training to wait at the next epoch, then continue |
| Stop training | You want the current run to halt |
| Resume from checkpoint | You raised the epoch count and want the same experiment to continue |
| Apply constraints | You want generated rows checked against the rules in the box |
| Settings | You want to change the privacy labels, or delete this session’s table or files |

Deleting files in Settings asks you to confirm first. It only removes files inside this project.

## What a finished run leaves on disk

| Folder | What is in it |
|---|---|
| `models/checkpoints/<experiment id>/` | The trained model |
| `data/synthetic/<experiment id>/` | The new table |
| `data/reports/<experiment id>/` | The HTML and PDF report |
| `artifacts/experiments/<experiment id>/` | The record of settings and results |

## If something looks wrong

- Refresh the browser if a page does not match the latest start of the app.
- Generate again after training. An older synthetic file is not updated by itself.
- A short demo can place values on the edge of the ranges in the original table. Use the quality page to see that. It does not mean the button failed.
- If every row is rejected after you turn constraints on, the rules are stricter than the generated values. Uncheck **Apply constraints**, or relax the rules, and generate again.
- Technical detail from a failure is written to `artifacts/logs/tabdiffuse.log`.
