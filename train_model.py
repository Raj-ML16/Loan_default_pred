# Clean & Structural ML Pipeline - Production Ready
# Focused on core ML functionality: Load → Prep → Train → Evaluate → Save

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Imbalanced Learning
try:
    from imblearn.over_sampling import SMOTE
    IMBALANCE_AVAILABLE = True
except ImportError:
    print("⚠️ Install: pip install imbalanced-learn")
    IMBALANCE_AVAILABLE = False

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ Install: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_score, recall_score, f1_score, accuracy_score
)
from collections import Counter


class LoanDefaultMLPipeline:
    """
    Clean, structural ML pipeline for loan default prediction
    Focus: Core functionality without complexity
    """
    
    def __init__(self, config=None):
        """Initialize pipeline with configuration"""
        # Core data attributes
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_processed = None
        self.X_test_processed = None
        
        # Pipeline components
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Configuration
        self.config = config or self._default_config()
        
        print("🚀 Loan Default ML Pipeline Initialized")
        print(f"📋 Configuration: {self.config}")
    
    def _default_config(self):
        """Default pipeline configuration"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'n_features': 20,
            'cv_folds': 3,
            'smote_ratio': 3.0,  # Conservative 3:1 instead of 1:1
            'regularization': 'moderate'  # light, moderate, heavy
        }
    
    def load_data(self, file_path='processed_loan_data.csv'):
        """Load and validate processed data"""
        print("\n" + "="*60)
        print("📁 LOADING DATA")
        print("="*60)
        
        try:
            # Load data
            data = pd.read_csv(file_path)
            print(f"✓ Data loaded: {data.shape}")
            
            # Validate target column
            if 'Default' not in data.columns:
                raise ValueError("❌ Target column 'Default' not found!")
            
            # Separate features and target
            X = data.drop('Default', axis=1)
            y = data['Default']
            
            # Basic validation
            print(f"✓ Features: {X.shape}")
            print(f"✓ Target distribution: {Counter(y)}")
            
            # Calculate imbalance ratio
            imbalance_ratio = (y == 0).sum() / (y == 1).sum()
            print(f"✓ Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 20:
                print("⚠️ Very high imbalance detected")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def prepare_data(self, X, y):
        """Prepare data: split, scale, select features"""
        print("\n" + "="*60)
        print("🔧 DATA PREPARATION")
        print("="*60)
        
        # 1. Train-test split (stratified)
        print("📊 Splitting data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        print(f"✓ Train set: {self.X_train.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        
        # 2. Feature scaling
        print("📏 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        print("✓ Features scaled")
        
        # 3. Feature selection
        print(f"🎯 Selecting top {self.config['n_features']} features...")
        self.feature_selector = SelectKBest(
            score_func=f_classif, 
            k=self.config['n_features']
        )
        
        self.X_train_processed = self.feature_selector.fit_transform(X_train_scaled, self.y_train)
        self.X_test_processed = self.feature_selector.transform(X_test_scaled)
        
        print(f"✓ Features reduced: {X_train_scaled.shape[1]} → {self.X_train_processed.shape[1]}")
        
        # Show class distribution
        train_dist = Counter(self.y_train)
        test_dist = Counter(self.y_test)
        print(f"✓ Train distribution: {train_dist}")
        print(f"✓ Test distribution: {test_dist}")
        
        return self.X_train_processed, self.X_test_processed
    
    def handle_imbalance(self, X_train, y_train):
        """Handle class imbalance with conservative SMOTE"""
        print("\n" + "="*60)
        print("⚖️ HANDLING CLASS IMBALANCE")
        print("="*60)
        
        if not IMBALANCE_AVAILABLE:
            print("⚠️ SMOTE not available, using original data")
            return X_train, y_train
        
        original_dist = Counter(y_train)
        original_ratio = original_dist[0] / original_dist[1]
        
        print(f"📊 Original distribution: {original_dist}")
        print(f"📊 Original ratio: {original_ratio:.2f}:1")
        
        # Conservative SMOTE - target ratio 3:1 instead of 1:1
        target_ratio = self.config['smote_ratio']
        target_minority = int(original_dist[0] / target_ratio)
        
        print(f"🎯 Target ratio: {target_ratio}:1")
        print(f"🎯 Target minority samples: {target_minority}")
        
        try:
            smote = SMOTE(
                sampling_strategy={1: target_minority},
                random_state=self.config['random_state'],
                k_neighbors=3  # Conservative
            )
            
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            new_dist = Counter(y_resampled)
            new_ratio = new_dist[0] / new_dist[1]
            
            print(f"✓ New distribution: {new_dist}")
            print(f"✓ New ratio: {new_ratio:.2f}:1")
            print(f"✓ Dataset size: {X_resampled.shape[0]} samples")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"❌ SMOTE failed: {e}")
            print("Using original data")
            return X_train, y_train
    
    def initialize_models(self):
        """Initialize regularized models"""
        print("\n" + "="*60)
        print("🤖 INITIALIZING MODELS")
        print("="*60)
        
        # Regularization levels
        reg_levels = {
            'light': {'C': 1.0, 'max_depth': 10, 'n_estimators': 100},
            'moderate': {'C': 0.1, 'max_depth': 8, 'n_estimators': 50},
            'heavy': {'C': 0.01, 'max_depth': 6, 'n_estimators': 30}
        }
        
        reg = reg_levels[self.config['regularization']]
        
        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.config['random_state'],
                max_iter=1000,
                class_weight='balanced',
                C=reg['C'],
                solver='liblinear'
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=reg['n_estimators'],
                max_depth=reg['max_depth'],
                random_state=self.config['random_state'],
                class_weight='balanced',
                min_samples_split=20,
                min_samples_leaf=10,
                n_jobs=1  # Avoid parallel issues
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=reg['n_estimators'],
                max_depth=reg['max_depth']-2,  # More conservative
                learning_rate=0.05,
                random_state=self.config['random_state'],
                scale_pos_weight=3,  # Conservative class weight
                reg_alpha=0.1,
                reg_lambda=1.0,
                verbosity=0
            )
        
        print(f"✓ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  • {name}")
        
        return self.models
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("🏋️ TRAINING MODELS")
        print("="*60)
        
        self.results = {}
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, 
                           random_state=self.config['random_state'])
        
        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"\n🔄 [{i}/{len(self.models)}] Training {name}...")
            
            try:
                # Cross-validation
                print("   📊 Cross-validation...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                # Train on full training set
                print("   🎯 Training on full dataset...")
                model.fit(X_train, y_train)
                
                # Test predictions
                print("   🔮 Making predictions...")
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else accuracy
                
                # Calculate overfitting gap
                cv_mean = cv_scores.mean()
                overfitting_gap = cv_mean - roc_auc
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_mean,
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'test_roc_auc': roc_auc,
                    'overfitting_gap': overfitting_gap,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # Status indicator
                gap_status = "🟢" if overfitting_gap < 0.05 else "🟡" if overfitting_gap < 0.15 else "🔴"
                
                print(f"✅ {name} completed!")
                print(f"   📊 CV ROC-AUC: {cv_mean:.4f} (±{cv_scores.std():.4f})")
                print(f"   📈 Test ROC-AUC: {roc_auc:.4f}")
                print(f"   🛡️ Overfitting: {overfitting_gap:.4f} {gap_status}")
                print(f"   ⚖️ Precision: {precision:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        print(f"\n🎉 Training completed! {len(self.results)}/{len(self.models)} models successful")
        return self.results
    
    def evaluate_models(self):
        """Compare and select best model"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        if not self.results:
            print("❌ No results to evaluate")
            return None
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'CV_ROC_AUC': result['cv_mean'],
                'Test_ROC_AUC': result['test_roc_auc'],
                'Overfitting_Gap': result['overfitting_gap'],
                'Test_Precision': result['test_precision'],
                'Test_Recall': result['test_recall'],
                'Test_F1': result['test_f1'],
                'Test_Accuracy': result['test_accuracy']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate composite score (lower overfitting gap is better)
        comparison_df['Composite_Score'] = (
            comparison_df['Test_ROC_AUC'] * 0.4 +
            comparison_df['Test_F1'] * 0.3 +
            comparison_df['Test_Precision'] * 0.2 -
            comparison_df['Overfitting_Gap'] * 0.1
        )
        
        # Sort by composite score
        comparison_df = comparison_df.sort_values('Composite_Score', ascending=False)
        
        print("🏆 MODEL COMPARISON:")
        print("=" * 90)
        display_cols = ['Model', 'CV_ROC_AUC', 'Test_ROC_AUC', 'Overfitting_Gap', 
                       'Test_Precision', 'Test_F1', 'Composite_Score']
        print(comparison_df[display_cols].round(4).to_string(index=False))
        
        # Select best model
        self.best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.results[self.best_model_name]['model']
        
        best_result = self.results[self.best_model_name]
        
        print(f"\n🥇 BEST MODEL: {self.best_model_name}")
        print(f"   🎯 Test ROC-AUC: {best_result['test_roc_auc']:.4f}")
        print(f"   🛡️ Overfitting Gap: {best_result['overfitting_gap']:.4f}")
        print(f"   ⚖️ Precision: {best_result['test_precision']:.4f}")
        print(f"   📈 F1-Score: {best_result['test_f1']:.4f}")
        
        # Generate classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print("-" * 50)
        print(classification_report(self.y_test, best_result['y_pred']))
        
        return comparison_df
    
    def save_model(self, filename=None):
        """Save the best model with metadata"""
        print("\n" + "="*60)
        print("💾 SAVING MODEL")
        print("="*60)
        
        if self.best_model is None:
            print("❌ No model to save")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loan_default_model_{timestamp}.pkl"
        
        try:
            # Prepare model package
            model_package = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'config': self.config,
                'results': self.results[self.best_model_name],
                'training_date': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Save model
            joblib.dump(model_package, filename)
            
            print(f"✅ Model saved successfully!")
            print(f"📁 Filename: {filename}")
            print(f"🤖 Model: {self.best_model_name}")
            print(f"📊 Test ROC-AUC: {self.results[self.best_model_name]['test_roc_auc']:.4f}")
            print(f"🛡️ Overfitting Gap: {self.results[self.best_model_name]['overfitting_gap']:.4f}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def create_visualizations(self):
        """Create essential visualizations"""
        print("\n" + "="*60)
        print("📊 CREATING VISUALIZATIONS")
        print("="*60)
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Model Performance Comparison
            ax1 = axes[0, 0]
            models = list(self.results.keys())
            cv_scores = [self.results[m]['cv_mean'] for m in models]
            test_scores = [self.results[m]['test_roc_auc'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, cv_scores, width, label='CV ROC-AUC', alpha=0.8)
            ax1.bar(x + width/2, test_scores, width, label='Test ROC-AUC', alpha=0.8)
            ax1.set_xlabel('Models')
            ax1.set_ylabel('ROC-AUC')
            ax1.set_title('🏆 Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.replace(' ', '\n') for m in models])
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # 2. Overfitting Analysis
            ax2 = axes[0, 1]
            gaps = [self.results[m]['overfitting_gap'] for m in models]
            colors = ['green' if gap < 0.05 else 'orange' if gap < 0.15 else 'red' for gap in gaps]
            
            bars = ax2.bar(models, gaps, color=colors, alpha=0.7)
            ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
            ax2.axhline(y=0.15, color='red', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Models')
            ax2.set_ylabel('CV-Test Gap')
            ax2.set_title('🛡️ Overfitting Analysis')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(alpha=0.3)
            
            # 3. ROC Curve (Best Model)
            ax3 = axes[1, 0]
            best_result = self.results[self.best_model_name]
            if best_result['y_pred_proba'] is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(self.y_test, best_result['y_pred_proba'])
                ax3.plot(fpr, tpr, linewidth=2, label=f'{self.best_model_name} (AUC={best_result["test_roc_auc"]:.3f})')
                ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax3.set_xlabel('False Positive Rate')
                ax3.set_ylabel('True Positive Rate')
                ax3.set_title(f'📈 ROC Curve - {self.best_model_name}')
                ax3.legend()
                ax3.grid(alpha=0.3)
            
            # 4. Confusion Matrix (Best Model)
            ax4 = axes[1, 1]
            cm = confusion_matrix(self.y_test, best_result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                       xticklabels=['No Default', 'Default'],
                       yticklabels=['No Default', 'Default'])
            ax4.set_title(f'🎯 Confusion Matrix - {self.best_model_name}')
            ax4.set_ylabel('True Label')
            ax4.set_xlabel('Predicted Label')
            
            plt.suptitle('🚀 Loan Default Prediction - Model Evaluation', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print("✅ Visualizations created successfully!")
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    def run_pipeline(self, data_file='processed_loan_data.csv'):
        """Run the complete ML pipeline"""
        print("🚀 STARTING ML PIPELINE")
        print("🎯 Clean & Structural Implementation")
        print("="*80)
        
        try:
            # Step 1: Load data
            X, y = self.load_data(data_file)
            if X is None or y is None:
                print("❌ Pipeline failed at data loading")
                return None
            
            # Step 2: Prepare data
            X_train_processed, X_test_processed = self.prepare_data(X, y)
            
            # Step 3: Handle imbalance
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train_processed, self.y_train)
            
            # Step 4: Initialize models
            self.initialize_models()
            
            # Step 5: Train models
            self.train_models(X_train_balanced, y_train_balanced, X_test_processed, self.y_test)
            
            # Step 6: Evaluate models
            comparison_df = self.evaluate_models()
            
            # Step 7: Save best model
            model_filename = self.save_model()
            
            # Step 8: Create visualizations
            self.create_visualizations()
            
            # Final summary
            print("\n" + "="*80)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("✅ Data processed and models trained")
            print("✅ Best model selected and saved")
            print("✅ Visualizations generated")
            print("✅ Ready for deployment")
            
            if self.best_model_name:
                best_result = self.results[self.best_model_name]
                print(f"\n🏆 FINAL RESULTS:")
                print(f"   🥇 Best Model: {self.best_model_name}")
                print(f"   📊 Test ROC-AUC: {best_result['test_roc_auc']:.4f}")
                print(f"   🛡️ Overfitting Gap: {best_result['overfitting_gap']:.4f}")
                print(f"   ⚖️ Precision: {best_result['test_precision']:.4f}")
                print(f"   📈 F1-Score: {best_result['test_f1']:.4f}")
                print(f"   💾 Model saved as: {model_filename}")
            
            return {
                'comparison_df': comparison_df,
                'best_model_name': self.best_model_name,
                'model_filename': model_filename,
                'results': self.results
            }
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# Configuration and Execution
def create_pipeline_config(regularization='moderate', n_features=20, cv_folds=3):
    """Create pipeline configuration"""
    return {
        'test_size': 0.2,
        'random_state': 42,
        'n_features': n_features,
        'cv_folds': cv_folds,
        'smote_ratio': 3.0,  # Conservative sampling
        'regularization': regularization  # light, moderate, heavy
    }


def main():
    """Main execution function"""
    print("🏭 LOAN DEFAULT PREDICTION - CLEAN ML PIPELINE")
    print("🎯 Structural Implementation - Core Functionality Only")
    print("="*80)
    
    # Create configuration
    config = create_pipeline_config(
        regularization='moderate',  # Prevents overfitting
        n_features=20,             # Reduced features
        cv_folds=3                 # Fast validation
    )
    
    # Initialize and run pipeline
    pipeline = LoanDefaultMLPipeline(config=config)
    results = pipeline.run_pipeline('processed_loan_data.csv')
    
    if results:
        print("\n🚀 Pipeline completed successfully!")
        print("📦 Ready for next steps: Deployment, Monitoring, etc.")
    else:
        print("\n❌ Pipeline failed - check errors above")


if __name__ == "__main__":
    main()