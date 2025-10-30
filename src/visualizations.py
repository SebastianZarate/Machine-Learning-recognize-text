"""Módulo de visualizaciones para análisis de sentimientos."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, Any

plt.style.use('ggplot')
sns.set_palette("husl")


def plot_confusion_matrices(confusion_matrices_dict, output_path='results/confusion_matrices.png'):
    """Genera matrices de confusión para múltiples modelos."""
    print("\n📊 Generando matrices de confusión...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (model_name, cm) in enumerate(confusion_matrices_dict.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=True)
        axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted', fontsize=10)
        axes[idx].set_ylabel('Actual', fontsize=10)
        axes[idx].set_xticklabels(['Negative', 'Positive'])
        axes[idx].set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Matrices guardadas: {output_path}")
    plt.show()
    plt.close()


def plot_roc_curves(roc_data_dict, output_path='results/roc_curves.png'):
    """Genera curvas ROC para múltiples modelos."""
    print("\n📈 Generando curvas ROC...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (model_name, roc_data) in enumerate(roc_data_dict.items()):
        plt.plot(roc_data['fpr'], roc_data['tpr'], color=colors[idx], lw=2, 
                label=f"{model_name} (AUC = {roc_data['auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Curvas ROC: {output_path}")
    plt.show()
    plt.close()


def plot_metrics_comparison(metrics_df, output_path='results/metrics_comparison.png'):
    """Genera comparación de métricas con barras agrupadas."""
    print("\n📊 Comparación de métricas...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_df))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        pos = x + (idx - 1.5) * width
        bars = ax.bar(pos, metrics_df[metric], width, label=label, color=colors[idx], alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.legend(loc='upper left')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparación: {output_path}")
    plt.show()
    plt.close()


def generate_wordclouds(df, output_path='results/wordclouds.png'):
    """Genera word clouds para reseñas positivas y negativas."""
    print("\n☁️  Generando word clouds...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pos_text = ' '.join(df[df['sentiment'] == 1]['review_clean'].astype(str))
    neg_text = ' '.join(df[df['sentiment'] == 0]['review_clean'].astype(str))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    wc_pos = WordCloud(width=800, height=400, background_color='white', max_words=100, 
                       colormap='Greens').generate(pos_text)
    axes[0].imshow(wc_pos, interpolation='bilinear')
    axes[0].set_title('Positive Reviews', fontsize=14, fontweight='bold', color='green')
    axes[0].axis('off')
    
    wc_neg = WordCloud(width=800, height=400, background_color='white', max_words=100, 
                       colormap='Reds').generate(neg_text)
    axes[1].imshow(wc_neg, interpolation='bilinear')
    axes[1].set_title('Negative Reviews', fontsize=14, fontweight='bold', color='red')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Word clouds: {output_path}")
    plt.show()
    plt.close()


def plot_feature_importance(model, vectorizer, top_n=20, output_path='results/feature_importance.png'):
    """Genera gráfico de características más importantes."""
    print("\n🔍 Importancia de características...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    features = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        imp = model.coef_[0]
        title = f'Top {top_n} Features (Logistic Regression)'
    elif hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        title = f'Top {top_n} Features (Random Forest)'
    else:
        print("⚠️  Modelo no soportado")
        return
    
    df = pd.DataFrame({'feature': features, 'importance': imp})
    df['abs_imp'] = df['importance'].abs()
    df = df.nlargest(top_n, 'abs_imp').sort_values('importance')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in df['importance']]
    ax.barh(range(len(df)), df['importance'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['feature'], fontsize=9)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Importancia: {output_path}")
    plt.show()
    plt.close()


def plot_prediction_distribution(y_test, predictions_dict, output_path='results/prediction_distribution.png'):
    """Genera histogramas de distribución de predicciones."""
    print("\n📊 Distribución de predicciones...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    unique, counts = np.unique(y_test, return_counts=True)
    real = dict(zip(unique, counts))
    
    for idx, (name, preds) in enumerate(predictions_dict.items()):
        u, c = np.unique(preds, return_counts=True)
        pred = dict(zip(u, c))
        
        cats = ['Negative', 'Positive']
        real_vals = [real.get(0, 0), real.get(1, 0)]
        pred_vals = [pred.get(0, 0), pred.get(1, 0)]
        
        x = np.arange(2)
        w = 0.35
        axes[idx].bar(x - w/2, real_vals, w, label='Actual', color='steelblue', alpha=0.8)
        axes[idx].bar(x + w/2, pred_vals, w, label='Predicted', color='coral', alpha=0.8)
        
        for i, (r, p) in enumerate(zip(real_vals, pred_vals)):
            axes[idx].text(i - w/2, r, str(r), ha='center', va='bottom', fontsize=9)
            axes[idx].text(i + w/2, p, str(p), ha='center', va='bottom', fontsize=9)
        
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(cats)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Prediction Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Distribución: {output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Módulo visualizations cargado ✅")
