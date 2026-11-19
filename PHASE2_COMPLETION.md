# Phase 2 Completion Checklist

Verified on 19 November 2026.

- [x] Dummy baseline implemented.
- [x] Logistic Regression, Multinomial Naive Bayes, and Linear SVM implemented.
- [x] Word and character TF-IDF features implemented.
- [x] Training-only validation used for model selection.
- [x] BiLSTM implemented with class weights and validation checkpoint selection.
- [x] Compact pretrained BERT fine-tuned for three sentiment classes.
- [x] Every final model evaluated on the same untouched 3,898-row test set.
- [x] Accuracy, macro-F1, weighted F1, per-class report, and confusion matrices produced.
- [x] Model comparison chart and `model_results.csv` produced.
- [x] Qualitative error-analysis sample exported.
- [x] Selected TF-IDF model, BiLSTM state, and BERT state saved.
- [x] Command-line prediction example added.
- [x] Working Streamlit website added in its own `app/` folder.
- [x] Website input-to-prediction flow passes an automated smoke test.
- [x] All seven notebooks executed with no error or unexecuted code cells.
- [x] Dependencies pass `pip check`.
- [x] Saved metrics recomputed from prediction files and matched exactly.
- [x] Saved models reload successfully.
- [x] All generated PNG files are valid.
- [x] No project artifact exceeds GitHub's 100 MB per-file limit.

The best model is Logistic Regression with word and character TF-IDF:

- Accuracy: **0.6811**
- Macro-F1: **0.6767**
- Weighted F1: **0.6815**

Run the automated verification again with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_phase2.py
```
