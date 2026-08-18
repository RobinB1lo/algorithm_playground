"""
Comprehensive GMDH Algorithm Comparison Framework

Loads environmental datasets from CSV files and compares all 13 GMDH variants
against standard ML baselines.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Import GMDH variants
from vanilla.gmdh import GMDH
from evaluation_metric_change.aic.gmdh_aic import GMDH_AIC
from evaluation_metric_change.hierarchical.hierarchical_gmdh import GMDH_Hierarchical
from cv_and_look_back.cv.cv_gmdh import GMDH_CV
from fractional_polynomials.cfp.cfp_gmdh import GMDH_Constrained
from fractional_polynomials.ufp.ufp_gmdh import GMDH_Unconstrained
from fractional_polynomials.ufp_hierarchical.ufp_hierarchical_gmdh import GMDH_UFP_Hierarchical
from fractional_polynomials.ufp_hierarchical_cv.ufp_hierarchical_cv_gmdh import GMDH_UFP_Hierarchical_CV
from basis_change.fourier.fourier_basis_gmdh import GMDH_Trig
from basis_change.radial.rbf_basis_gmdh import GMDH_RBF
from basis_change.sigmoid.sigmoid_basis_gmdh import GMDH_Sigmoid
from basis_change.spline.spline_basis_gmdh import GMDH_Spline
from cv_and_look_back.look_back.look_back_gmdh import GMDH_LookBack

warnings.filterwarnings('ignore')


class EnvironmentalDataLoader:
    """Load and manage environmental datasets from CSV files."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        else:
            data_dir = Path(data_dir)
        
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Please run: python data_generation.py"
            )
        self.data_dir = data_dir
    
    def load_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load dataset by name.
        
        Returns:
            X: Feature matrix
            y: Target vector
            description: Dataset description
        """
        csv_file = self.data_dir / f"{name}.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        # Last column is target
        X = df.iloc[:, :-1].values.astype(np.float64)
        y = df.iloc[:, -1].values.astype(np.float64)
        
        target_name = df.columns[-1]
        n_samples, n_features = X.shape
        ratio = n_samples / n_features
        
        description = (
            f"{name.capitalize()} | "
            f"Shape: {X.shape} | "
            f"Ratio (S:F): {ratio:.1f}:1"
        )
        
        return X, y, description


class GMDHComparison:
    """Comprehensive comparison framework for GMDH variants."""
    
    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.results = []
        
        # Initialize all 13 GMDH variants
        self.gmdh_models = {
            'Vanilla GMDH': GMDH(random_state=random_state, max_layers=15, ridge=1e-3, n_keep=8),
            'AIC': GMDH_AIC(random_state=random_state, max_layers=15, ridge=1e-3, n_keep=8),
            'Hierarchical': GMDH_Hierarchical(random_state=random_state, max_layers=15, ridge=1e-3, n_keep=8),
            'CV': GMDH_CV(random_state=random_state, max_layers=15, n_keep=8, k_folds=5),
            'Constrained Fractional Poly': GMDH_Constrained(random_state=random_state, max_layers=15, n_keep=8),
            'Unconstrained Fractional Poly': GMDH_Unconstrained(random_state=random_state, max_layers=15, n_keep=8),
            'UFP Hierarchical': GMDH_UFP_Hierarchical(random_state=random_state, max_layers=15, n_keep=8),
            'UFP Hierarchical CV': GMDH_UFP_Hierarchical_CV(random_state=random_state, max_layers=15, n_keep=8, k_folds=5),
            'Fourier/Trig': GMDH_Trig(random_state=random_state, max_layers=15, n_keep=8),
            'RBF': GMDH_RBF(random_state=random_state, max_layers=15, n_keep=8),
            'Sigmoid': GMDH_Sigmoid(random_state=random_state, max_layers=15, n_keep=8),
            'Spline': GMDH_Spline(random_state=random_state, max_layers=15, n_keep=8),
            'LookBack': GMDH_LookBack(random_state=random_state, max_layers=15, n_keep=8),
        }
        
        # Baseline models
        self.baseline_models = {
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state, max_depth=5),
        }
    
    def evaluate_model(self, model, X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray, model_name: str,
                      timeout: int = 300) -> Dict:
        """Train and evaluate a model with timeout."""
        start_time = time.time()
        result = {
            'model': model_name,
            'train_r2': np.nan,
            'test_r2': np.nan,
            'train_rmse': np.nan,
            'test_rmse': np.nan,
            'train_mae': np.nan,
            'test_mae': np.nan,
            'time': np.nan,
            'status': 'pending'
        }
        
        try:
            # Train model
            model.fit(X_train, y_train)
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                result['status'] = f'timeout ({elapsed:.1f}s)'
                return result
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            result['train_r2'] = r2_score(y_train, y_train_pred)
            result['test_r2'] = r2_score(y_test, y_test_pred)
            result['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
            result['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
            result['train_mae'] = mean_absolute_error(y_train, y_train_pred)
            result['test_mae'] = mean_absolute_error(y_test, y_test_pred)
            result['time'] = elapsed
            result['status'] = 'success'
            
        except Exception as e:
            result['status'] = f'failed: {str(e)[:40]}'
            result['time'] = time.time() - start_time
        
        return result
    
    def compare_on_dataset(self, X: np.ndarray, y: np.ndarray, dataset_name: str, description: str):
        """Run full comparison on a single dataset."""
        print("\n" + "="*80)
        print(f"DATASET: {description}")
        print("="*80)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Train/test split: {len(X_train)}/{len(X_test)}")
        print(f"\nTraining models...")
        
        dataset_results = []
        
        # GMDH models
        print("\n--- GMDH Variants (13) ---")
        for name, model in self.gmdh_models.items():
            result = self.evaluate_model(model, X_train, X_test, y_train, y_test, name)
            dataset_results.append(result)
            
            status_icon = "✓" if result['status'] == 'success' else "✗"
            print(f"{status_icon} {name:30} | "
                  f"Test R²: {result['test_r2']:7.4f} | "
                  f"Time: {result['time']:6.3f}s")
            
            if result['status'] != 'success':
                print(f"  └─ {result['status']}")
        
        # Baseline models
        print("\n--- Baseline Models ---")
        for name, model in self.baseline_models.items():
            result = self.evaluate_model(model, X_train, X_test, y_train, y_test, name)
            dataset_results.append(result)
            
            status_icon = "✓" if result['status'] == 'success' else "✗"
            print(f"{status_icon} {name:30} | "
                  f"Test R²: {result['test_r2']:7.4f} | "
                  f"Time: {result['time']:6.3f}s")
        
        self.results.extend([(dataset_name, r) for r in dataset_results])
    
    def plot_results(self, output_dir: str = None):
        """Generate comparison plots."""
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Convert results to DataFrame
        results_list = [
            {**r, 'dataset': d} for d, r in self.results
        ]
        df_results = pd.DataFrame(results_list)

        # Only plot successful models
        df_plot = df_results[df_results['status'] == 'success'].copy()

        if len(df_plot) == 0:
            print("No successful results to plot.")
            return

        # 1. Test R² by dataset
        datasets = sorted(df_plot['dataset'].unique())
        n_datasets = len(datasets)

        # Adjust figure width based on number of datasets
        fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
        # Ensure axes is always a 1D array
        if n_datasets == 1:
            axes = [axes]

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            data = df_plot[df_plot['dataset'] == dataset].sort_values('test_r2', ascending=True)

            colors = ['#378ADD' if 'Vanilla' in m or any(x in m for x in ['AIC', 'Hierarchical', 'CV', 'UFP', 'Fourier', 'RBF', 'Sigmoid', 'Spline', 'LookBack', 'Constrained', 'Unconstrained']) else '#185FA5' for m in data['model']]
            ax.barh(data['model'], data['test_r2'], color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Test R²', fontsize=11, fontweight=500)
            ax.set_title(f'{dataset.capitalize()}', fontsize=12, fontweight=500)
            ax.set_xlim([-0.05, 1.0])
            ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_dir / "test_r2_comparison.png", dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: test_r2_comparison.png")

        # 2. Computation time
        fig, ax = plt.subplots(figsize=(12, 6))
        data = df_plot.sort_values('time', ascending=True)
        colors = ['#378ADD' if any(x in m for x in ['GMDH', 'AIC', 'Hierarchical', 'CV', 'UFP', 'Fourier', 'RBF', 'Sigmoid', 'Spline', 'LookBack', 'Constrained', 'Unconstrained']) else '#185FA5' for m in data['model']]
        ax.barh(data['model'], data['time'], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Time (seconds)', fontsize=11, fontweight=500)
        ax.set_title('Computation Time per Model', fontsize=12, fontweight=500)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_dir / "computation_time.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved: computation_time.png")

        # 3. Performance summary table
        summary = df_plot.groupby('model')[['test_r2', 'test_rmse', 'test_mae', 'time']].mean().sort_values('test_r2', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('tight')
        ax.axis('off')

        table_data = []
        for idx, (model, row) in enumerate(summary.iterrows()):
            table_data.append([
                model,
                f"{row['test_r2']:.4f}",
                f"{row['test_rmse']:.4f}",
                f"{row['test_mae']:.4f}",
                f"{row['time']:.3f}s"
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=['Model', 'Avg Test R²', 'Avg RMSE', 'Avg MAE', 'Avg Time'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Header styling
        for i in range(5):
            table[(0, i)].set_facecolor('#185FA5')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(5):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

        plt.savefig(output_dir / "performance_summary.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved: performance_summary.png")

        print(f"\nAll plots saved to: {output_dir}")

    def export_results_csv(self, output_dir: str = None):
        """
        Export the results to a CSV file.
        Columns: dataset, model, test_r2, test_rmse, test_mae, time, status.
        This matches the R script's output format for easy comparison.
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Build list of rows
        rows = []
        for dataset_name, result in self.results:
            row = {
                'dataset': dataset_name,
                'model': result['model'],
                'test_r2': result['test_r2'],
                'test_rmse': result['test_rmse'],
                'test_mae': result['test_mae'],
                'time': result['time'],
                'status': result['status']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = output_dir / "python_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved results CSV: {csv_path}")


def main():
    """Run full comparison on all environmental datasets."""
    
    # Load data
    try:
        loader = EnvironmentalDataLoader()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nGenerating datasets first...")
        os.system("python data_generation.py")
        loader = EnvironmentalDataLoader()
    
    # Initialize comparison
    comp = GMDHComparison(random_state=42)
    
    # Run on each dataset
    datasets = [
        ('wildfire', 'Wildfire'),
        ('weather', 'Weather'),
        ('ecological', 'Ecological'),
        ('air_quality', 'Air Quality'),
        ('low_dim', 'Low-Dimensional Synthetic')
    ]
    
    for dataset_key, dataset_name in datasets:
        try:
            X, y, description = loader.load_dataset(dataset_key)
            comp.compare_on_dataset(X, y, dataset_key, description)
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate plots
    comp.plot_results()
    
    # Export CSV
    comp.export_results_csv()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()