import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def save_results(subject_id, best_val_loss, best_val_accuracy, test_accuracy, precision_sess1, recall_sess1, f1_sess1, confusion_matrix_sess1,
                 test_accuracy_5, precision_sess5, recall_sess5, f1_sess5, confusion_matrix_sess5):
    results_df = pd.DataFrame({
        "Subject": [subject_id],
        "Best Val Loss": [best_val_loss],
        "Best Val Accuracy": [best_val_accuracy],
        "Session 1 Test Accuracy": [test_accuracy],
        "Session 1 Precision": [precision_sess1],
        "Session 1 Recall": [recall_sess1],
        "Session 1 F1": [f1_sess1],
        "Session 5 Test Accuracy": [test_accuracy_5],
        "Session 5 Precision": [precision_sess5],
        "Session 5 Recall": [recall_sess5],
        "Session 5 F1": [f1_sess5]
    })
    results_df.to_csv(f'subject_{subject_id}_results.csv', index=False)

def plot_confusion_matrix(subject_id, cm):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for Subject {subject_id}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'subject_{subject_id}_confusion_matrix.png')
    plt.close()
