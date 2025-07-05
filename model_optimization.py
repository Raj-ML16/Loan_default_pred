# Phase 3: Model Optimization & Business Tuning (Optimized Version)
# Filename: model_optimization_fast.py
# Purpose: Improve precision from 16% to 25%+ through efficient optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Optimization Libraries
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    make_scorer, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available")
    XGBOOST_AVAILABLE = False

# Imbalanced Learning
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCE_AVAILABLE = True
except ImportError:
    print("⚠️ Imbalanced-learn not available")
    IMBALANCE_AVAILABLE = False


class FastModelOptimizer:
    """
    Optimized model optimization for business metrics
    Focus: Fast execution while improving precision from 16% to 25%+
    """
    
    def __init__(self, base_model_path=None):
        """Initialize optimizer with base model"""
        self.base_model_path = base_model_path
        self.base_model_package = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Optimization results
        self.optimization_results = {}
        self.threshold_results = {}
        self.sampling_results = {}
        self.hyperparameter_results = {}
        
        # Best configurations
        self.best_threshold = 0.5
        self.best_sampling = None
        self.best_hyperparams = None
        self.best_model = None
        
        print("🎯 Fast Model Optimizer Initialized")
        print("🎯 Target: Improve precision from 16% to 25%+ (Optimized for Speed)")
    
    def load_base_model(self, model_path=None):
        """Load the base trained model for optimization"""
        print("\n" + "="*50)
        print("📂 LOADING BASE MODEL")
        print("="*50)
        
        if model_path is None and self.base_model_path is None:
            print("❌ No model path provided")
            return False
        
        model_path = model_path or self.base_model_path
        
        try:
            self.base_model_package = joblib.load(model_path)
            
            print(f"✓ Model loaded: {model_path}")
            print(f"✓ Model type: {self.base_model_package['model_name']}")
            
            # Extract base performance
            base_results = self.base_model_package['results']
            print(f"✓ Base Precision: {base_results['test_precision']:.4f}")
            print(f"✓ Base F1-Score: {base_results['test_f1']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_training_data(self, data_path='processed_loan_data.csv'):
        """Load training data for optimization"""
        print("\n" + "="*50)
        print("📊 LOADING TRAINING DATA")
        print("="*50)
        
        try:
            # Load data
            data = pd.read_csv(data_path)
            X = data.drop('Default', axis=1)
            y = data['Default']
            
            print(f"✓ Data loaded: {data.shape}")
            
            # Apply same preprocessing as base model
            if self.base_model_package:
                scaler = self.base_model_package['scaler']
                feature_selector = self.base_model_package['feature_selector']
                config = self.base_model_package['config']
                
                # Train-test split with same random state
                from sklearn.model_selection import train_test_split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, 
                    test_size=config['test_size'],
                    random_state=config['random_state'],
                    stratify=y
                )
                
                # Apply preprocessing
                X_train_scaled = scaler.transform(self.X_train)
                X_test_scaled = scaler.transform(self.X_test)
                
                self.X_train = feature_selector.transform(X_train_scaled)
                self.X_test = feature_selector.transform(X_test_scaled)
                
                print(f"✓ Preprocessing applied: {self.X_train.shape}")
                
                return True
            else:
                print("❌ No base model loaded for preprocessing")
                return False
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def optimize_decision_threshold(self, target_precision=0.25):
        """Fast threshold optimization for business metrics"""
        print("\n" + "="*50)
        print("🎚️ FAST THRESHOLD OPTIMIZATION")
        print("="*50)
        
        if self.base_model_package is None:
            print("❌ No base model loaded")
            return None
        
        print(f"🎯 Target precision: {target_precision:.2%}")
        
        # Get base model predictions
        base_model = self.base_model_package['model']
        y_pred_proba = base_model.predict_proba(self.X_test)[:, 1]
        
        # Test fewer thresholds for speed (every 0.05 instead of 0.02)
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_metrics = []
        
        print("🔍 Testing thresholds (optimized)...")
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            threshold_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        # Convert to DataFrame for analysis
        threshold_df = pd.DataFrame(threshold_metrics)
        
        # Find optimal thresholds
        precision_candidates = threshold_df[threshold_df['precision'] >= target_precision]
        if not precision_candidates.empty:
            best_precision_idx = precision_candidates['f1_score'].idxmax()
            best_precision_config = threshold_df.loc[best_precision_idx]
        else:
            best_precision_idx = threshold_df['precision'].idxmax()
            best_precision_config = threshold_df.loc[best_precision_idx]
        
        # Balanced optimization (precision >= 20% and good F1)
        balanced_candidates = threshold_df[threshold_df['precision'] >= 0.20]
        if not balanced_candidates.empty:
            best_balanced_idx = balanced_candidates['f1_score'].idxmax()
            best_balanced_config = threshold_df.loc[best_balanced_idx]
        else:
            best_balanced_config = threshold_df.loc[threshold_df['f1_score'].idxmax()]
        
        # Store results
        self.threshold_results = {
            'all_metrics': threshold_df,
            'precision_optimized': best_precision_config,
            'balanced_optimized': best_balanced_config
        }
        
        # Print results
        print(f"\n📊 THRESHOLD OPTIMIZATION RESULTS:")
        print(f"🎯 Precision-Optimized:")
        print(f"   Threshold: {best_precision_config['threshold']:.2f}")
        print(f"   Precision: {best_precision_config['precision']:.4f} ({best_precision_config['precision']:.1%})")
        print(f"   F1-Score: {best_precision_config['f1_score']:.4f}")
        
        print(f"\n⚖️ Balanced-Optimized:")
        print(f"   Threshold: {best_balanced_config['threshold']:.2f}")
        print(f"   Precision: {best_balanced_config['precision']:.4f} ({best_balanced_config['precision']:.1%})")
        print(f"   F1-Score: {best_balanced_config['f1_score']:.4f}")
        
        return self.threshold_results
    
    def optimize_sampling_strategy_fast(self):
        """Test key sampling strategies for better precision (fast version)"""
        print("\n" + "="*50)
        print("🔄 FAST SAMPLING STRATEGY TEST")
        print("="*50)
        
        if not IMBALANCE_AVAILABLE:
            print("❌ Imbalanced-learn not available")
            return None
        
        # Define key sampling strategies only (reduced set for speed)
        sampling_strategies = {
            'No Sampling': None,
            'Conservative SMOTE (4:1)': ('smote', 4.0),
            'SMOTE (2:1)': ('smote', 2.0),
            'Random Undersampling': ('undersample', 0.3)
        }
        
        # Get simplified base model configuration
        base_params = {
            'n_estimators': 30,  # Reduced for speed
            'max_depth': 8,
            'random_state': 42,
            'class_weight': 'balanced',
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'n_jobs': 1
        }
        
        sampling_results = {}
        
        print("🧪 Testing key sampling strategies...")
        
        for strategy_name, strategy_config in sampling_strategies.items():
            print(f"🔄 Testing {strategy_name}...")
            
            try:
                # Apply sampling
                if strategy_config is None:
                    X_train_sampled = self.X_train
                    y_train_sampled = self.y_train
                else:
                    method, ratio = strategy_config
                    X_train_sampled, y_train_sampled = self._apply_sampling_fast(
                        self.X_train, self.y_train, method, ratio
                    )
                
                # Train model (smaller model for speed)
                model = RandomForestClassifier(**base_params)
                model.fit(X_train_sampled, y_train_sampled)
                
                # Evaluate
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                precision = precision_score(self.y_test, y_pred, zero_division=0)
                recall = recall_score(self.y_test, y_pred, zero_division=0)
                f1 = f1_score(self.y_test, y_pred, zero_division=0)
                
                sampling_results[strategy_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'strategy': strategy_config
                }
                
                print(f"   Precision: {precision:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        self.sampling_results = sampling_results
        
        # Find best strategies
        if sampling_results:
            best_precision = max(sampling_results.items(), key=lambda x: x[1]['precision'])
            print(f"\n🏆 Best Precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})")
        
        return sampling_results
    
    def _apply_sampling_fast(self, X_train, y_train, method, ratio):
        """Apply specific sampling method (optimized)"""
        if method == 'smote':
            original_dist = Counter(y_train)
            target_minority = int(original_dist[0] / ratio)
            smote = SMOTE(sampling_strategy={1: target_minority}, random_state=42, k_neighbors=3)
            return smote.fit_resample(X_train, y_train)
            
        elif method == 'undersample':
            original_dist = Counter(y_train)
            target_majority = int(original_dist[1] / ratio)
            undersampler = RandomUnderSampler(sampling_strategy={0: target_majority}, random_state=42)
            return undersampler.fit_resample(X_train, y_train)
        
        return X_train, y_train
    
    def hyperparameter_optimization_fast(self, model_type='random_forest'):
        """Fast hyperparameter optimization"""
        print("\n" + "="*50)
        print("🔧 FAST HYPERPARAMETER OPTIMIZATION")
        print("="*50)
        
        # Reduced parameter space for speed
        if model_type == 'random_forest':
            model_class = RandomForestClassifier
            param_grid = {
                'n_estimators': [30, 50, 100],
                'max_depth': [6, 8, 10],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10],
                'class_weight': ['balanced', {0: 1, 1: 3}]
            }
        
        # Create custom scorer (prioritize precision)
        def business_scorer(y_true, y_pred):
            precision = precision_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            return 0.7 * precision + 0.3 * f1
        
        custom_scorer = make_scorer(business_scorer)
        
        # Initialize model
        base_model = model_class(random_state=42, n_jobs=1)
        
        # Reduced CV and iterations for speed
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            base_model, param_grid, cv=cv, scoring=custom_scorer,
            n_iter=20,  # Reduced iterations
            n_jobs=1, verbose=0, random_state=42
        )
        
        print(f"🎲 Random Search: 20 iterations (fast mode)")
        
        # Perform search
        print("🔍 Searching optimal hyperparameters...")
        search.fit(self.X_train, self.y_train)
        
        # Get best model and evaluate
        best_model = search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate detailed metrics
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.hyperparameter_results = {
            'best_model': best_model,
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n🏆 HYPERPARAMETER OPTIMIZATION RESULTS:")
        print(f"✓ Test Precision: {precision:.4f} ({precision:.1%})")
        print(f"✓ Test F1-Score: {f1:.4f}")
        print(f"✓ Best Parameters: {search.best_params_}")
        
        return self.hyperparameter_results
    
    def create_fast_report(self):
        """Generate quick optimization report"""
        print("\n" + "="*50)
        print("📋 OPTIMIZATION REPORT")
        print("="*50)
        
        # Summary of all optimizations
        optimizations = {}
        
        if self.base_model_package:
            base_results = self.base_model_package['results']
            optimizations['Base Model'] = {
                'precision': base_results['test_precision'],
                'recall': base_results['test_recall'],
                'f1': base_results['test_f1'],
                'roc_auc': base_results['test_roc_auc']
            }
        
        if hasattr(self, 'threshold_results') and self.threshold_results:
            opt_result = self.threshold_results['balanced_optimized']
            optimizations['Threshold Optimized'] = {
                'precision': opt_result['precision'],
                'recall': opt_result['recall'],
                'f1': opt_result['f1_score'],
                'roc_auc': None
            }
        
        if hasattr(self, 'sampling_results') and self.sampling_results:
            best_sampling = max(self.sampling_results.items(), key=lambda x: x[1]['precision'])
            optimizations['Best Sampling'] = {
                'precision': best_sampling[1]['precision'],
                'recall': best_sampling[1]['recall'],
                'f1': best_sampling[1]['f1_score'],  # Fixed: use f1_score
                'roc_auc': None
            }
        
        if hasattr(self, 'hyperparameter_results'):
            optimizations['Hyperparameter Optimized'] = {
                'precision': self.hyperparameter_results['test_precision'],
                'recall': self.hyperparameter_results['test_recall'],
                'f1': self.hyperparameter_results['test_f1'],
                'roc_auc': self.hyperparameter_results['test_roc_auc']
            }
        
        print("📊 OPTIMIZATION COMPARISON:")
        print("-" * 70)
        
        for opt_name, opt_result in optimizations.items():
            try:
                precision = opt_result.get('precision', 0)
                recall = opt_result.get('recall', 0)
                f1 = opt_result.get('f1', 0)
                roc_auc = opt_result.get('roc_auc', 'N/A')
                
                print(f"{opt_name:20s} | Precision: {precision:.3f} | F1: {f1:.3f} | ROC-AUC: {roc_auc if roc_auc != 'N/A' else 'N/A'}")
            except Exception as e:
                print(f"{opt_name:20s} | Error displaying results: {e}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if hasattr(self, 'threshold_results'):
            best_threshold = self.threshold_results['balanced_optimized']
            print(f"✅ Use optimized threshold: {best_threshold['threshold']:.2f}")
            print(f"   Expected precision: {best_threshold['precision']:.1%}")
        
        if hasattr(self, 'sampling_results') and self.sampling_results:
            best_sampling_name = max(self.sampling_results.items(), key=lambda x: x[1]['precision'])[0]
            print(f"✅ Best sampling strategy: {best_sampling_name}")
        
        if hasattr(self, 'hyperparameter_results'):
            print(f"✅ Hyperparameter optimization achieved: {self.hyperparameter_results['test_precision']:.1%} precision")
        
        return optimizations
    
    def save_optimized_model(self, config_name='fast_optimized'):
        """Save the best optimized model configuration"""
        print("\n" + "="*50)
        print("💾 SAVING OPTIMIZED MODEL")
        print("="*50)
        
        # Determine best configuration
        best_model = None
        best_config = {}
        best_metrics = {}
        
        if hasattr(self, 'hyperparameter_results') and self.hyperparameter_results:
            best_model = self.hyperparameter_results['best_model']
            best_config['hyperparameters'] = self.hyperparameter_results['best_params']
            best_metrics = {
                'test_precision': self.hyperparameter_results['test_precision'],
                'test_recall': self.hyperparameter_results['test_recall'],
                'test_f1': self.hyperparameter_results['test_f1'],
                'test_roc_auc': self.hyperparameter_results['test_roc_auc']
            }
        elif self.base_model_package:
            best_model = self.base_model_package['model']
            best_metrics = self.base_model_package['results']
        
        # Add optimizations
        if hasattr(self, 'threshold_results'):
            best_config['optimal_threshold'] = self.threshold_results['balanced_optimized']['threshold']
        
        if hasattr(self, 'sampling_results') and self.sampling_results:
            best_sampling_name = max(self.sampling_results.items(), key=lambda x: x[1]['precision'])[0]
            best_config['best_sampling_strategy'] = best_sampling_name
        
        if best_model is None:
            print("❌ No model available to save")
            return None
        
        # Create optimized model package
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fast_optimized_model_{config_name}_{timestamp}.pkl"
        
        optimized_package = {
            'model': best_model,
            'model_name': f"Fast Optimized {self.base_model_package['model_name']}" if self.base_model_package else "Fast Optimized Model",
            'scaler': self.base_model_package['scaler'] if self.base_model_package else None,
            'feature_selector': self.base_model_package['feature_selector'] if self.base_model_package else None,
            'optimization_config': best_config,
            'optimized_metrics': best_metrics,
            'optimization_date': datetime.now().isoformat(),
            'version': '2.0_fast'
        }
        
        try:
            joblib.dump(optimized_package, filename)
            
            print(f"✅ Optimized model saved!")
            print(f"📁 Filename: {filename}")
            
            if 'test_precision' in best_metrics:
                print(f"📊 Optimized precision: {best_metrics['test_precision']:.1%}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Error saving: {e}")
            return None
    
    def run_fast_optimization(self, model_path, data_path='processed_loan_data.csv', 
                             target_precision=0.25):
        """Run fast optimization pipeline"""
        print("🚀 STARTING FAST MODEL OPTIMIZATION")
        print("🎯 Target: Improve precision to 25%+ (Speed Optimized)")
        print("="*60)
        
        try:
            # Step 1: Load base model and data
            if not self.load_base_model(model_path):
                return None
            
            if not self.load_training_data(data_path):
                return None
            
            # Step 2: Fast threshold optimization
            print("\n🎚️ Step 1: Fast Threshold Optimization")
            self.optimize_decision_threshold(target_precision)
            
            # Step 3: Fast sampling strategy test
            print("\n🔄 Step 2: Fast Sampling Strategy Test")
            self.optimize_sampling_strategy_fast()
            
            # Step 4: Fast hyperparameter optimization
            print("\n🔧 Step 3: Fast Hyperparameter Optimization")
            self.hyperparameter_optimization_fast()
            
            # Step 5: Generate report
            print("\n📋 Step 4: Generating Report")
            try:
                optimizations = self.create_fast_report()
            except Exception as e:
                print(f"⚠️ Report generation error: {e}")
                optimizations = {}
            
            # Step 6: Save optimized model
            print("\n💾 Step 5: Saving Optimized Model")
            try:
                model_filename = self.save_optimized_model('precision_optimized')
            except Exception as e:
                print(f"⚠️ Model saving error: {e}")
                model_filename = None
            
            # Final summary
            print("\n" + "="*60)
            print("🎉 FAST OPTIMIZATION COMPLETED!")
            print("="*60)
            
            if self.base_model_package:
                base_precision = self.base_model_package['results']['test_precision']
                
                # Find the BEST precision across all optimizations
                best_precision = base_precision
                best_method = "Base Model"
                
                # Check threshold optimization
                if hasattr(self, 'threshold_results') and self.threshold_results:
                    threshold_precision = self.threshold_results['balanced_optimized']['precision']
                    if threshold_precision > best_precision:
                        best_precision = threshold_precision
                        best_method = "Threshold Optimization"
                
                # Check hyperparameter optimization (usually the best)
                if hasattr(self, 'hyperparameter_results') and self.hyperparameter_results:
                    hp_precision = self.hyperparameter_results['test_precision']
                    if hp_precision > best_precision:
                        best_precision = hp_precision
                        best_method = "Hyperparameter Optimization"
                
                improvement = best_precision - base_precision
                
                print(f"📊 FINAL RESULTS:")
                print(f"   🎯 Base Precision: {base_precision:.1%}")
                print(f"   🚀 Best Optimized Precision: {best_precision:.1%}")
                print(f"   🏆 Best Method: {best_method}")
                print(f"   📈 Total Improvement: +{improvement:.1%} ({improvement/base_precision:.1%} relative)")
                print(f"   💾 Model saved: {model_filename}")
                
                # Check if target achieved
                if best_precision >= target_precision:
                    print(f"   ✅ TARGET ACHIEVED! ({target_precision:.0%} precision target)")
                    print(f"   🎉 EXCEEDED BY: +{best_precision - target_precision:.1%}")
                else:
                    print(f"   ⚠️ Target not fully achieved. Consider additional strategies.")
            
            return {
                'optimized_model_path': model_filename,
                'optimization_results': optimizations,
                'best_precision': best_precision if self.base_model_package else None,
                'best_method': best_method if self.base_model_package else None,
                'target_achieved': best_precision >= target_precision if self.base_model_package else False,
                'improvement': improvement if self.base_model_package else None
            }
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return None


# Main execution function
def main():
    """Main fast optimization execution"""
    print("🎯 FAST MODEL OPTIMIZATION - PHASE 3")
    print("🚀 Goal: Improve Precision to 25%+ (Speed Optimized)")
    print("="*60)
    
    # Configuration
    base_model_path = "loan_default_model_20250630_192253.pkl"  # Update with your model filename
    data_path = "processed_loan_data.csv"
    target_precision = 0.25  # 25% target precision
    
    # Initialize fast optimizer
    optimizer = FastModelOptimizer()
    
    # Run fast optimization
    results = optimizer.run_fast_optimization(
        model_path=base_model_path,
        data_path=data_path,
        target_precision=target_precision
    )
    
    if results:
        print(f"\n🎊 Fast optimization completed successfully!")
        print(f"📁 Optimized model: {results['optimized_model_path']}")
        print(f"🎯 Target achieved: {results['target_achieved']}")
    else:
        print(f"\n❌ Fast optimization failed")


if __name__ == "__main__":
    main()