# FIXED & STABLE Loan Default Prediction Pipeline
# This version fixes all the identified issues

import sys
import os

# Fix PyArrow issues by setting environment variables before imports
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')
    print("✓ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Try: pip install pandas==2.1.4 pyarrow==14.0.1")
    sys.exit(1)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class LoanDefaultMLPipeline:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load the dataset and fix data types immediately"""
        try:
            print("Loading data...")
            self.data = pd.read_csv(file_path, low_memory=False)
            print(f"✓ Data loaded successfully! Shape: {self.data.shape}")
            
            # Immediately fix data types
            self._fix_data_types()
            
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def _fix_data_types(self):
        """Fix data types for columns that should be numeric"""
        if self.data is None:
            return
        
        print("Fixing data types...")
        
        # Define columns that should be numeric
        numeric_columns = [
            'Client_Income', 'Credit_Amount', 'Loan_Annuity', 'Population_Region_Relative',
            'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days', 'Score_Source_3'
        ]
        
        # Convert string columns to numeric
        conversion_count = 0
        for col in numeric_columns:
            if col in self.data.columns:
                original_dtype = self.data[col].dtype
                if original_dtype == 'object':  # String type
                    try:
                        self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                        conversion_count += 1
                        print(f"  ✓ Converted {col} from {original_dtype} to numeric")
                    except Exception as e:
                        print(f"  ✗ Failed to convert {col}: {e}")
        
        print(f"✓ Data type conversion completed! ({conversion_count} columns converted)")
    
    def initial_data_exploration(self):
        """Perform initial data exploration"""
        if self.data is None:
            print("Please load data first")
            return
        
        print("="*60)
        print("INITIAL DATA EXPLORATION")
        print("="*60)
        
        # Basic info
        print(f"Dataset Shape: {self.data.shape}")
        print(f"Total Features: {self.data.shape[1]}")
        print(f"Total Samples: {self.data.shape[0]}")
        
        # Data types summary
        print(f"\nData Types Summary:")
        dtype_summary = self.data.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"  {dtype}: {count} columns")
        
        # Missing values analysis (top 10 only)
        print(f"\nTop 10 Columns with Missing Values:")
        missing_vals = self.data.isnull().sum()
        missing_percent = (missing_vals / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_vals,
            'Missing_Percentage': missing_percent
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.head(10).to_string())
        else:
            print("  No missing values found!")
        
        # Target variable distribution
        if 'Default' in self.data.columns:
            print(f"\nTarget Variable Distribution:")
            target_dist = self.data['Default'].value_counts()
            target_percent = self.data['Default'].value_counts(normalize=True) * 100
            print(f"  No Default (0): {target_dist[0]} ({target_percent[0]:.2f}%)")
            print(f"  Default (1): {target_dist[1]} ({target_percent[1]:.2f}%)")
            
            # Check for class imbalance
            imbalance_ratio = target_dist[0] / target_dist[1]
            print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 3:
                print(f"  ⚠️  Dataset is imbalanced - will need SMOTE/balancing")
        
        return missing_df
    
    def data_quality_check(self):
        """Perform comprehensive data quality checks"""
        if self.data is None:
            print("Please load data first")
            return
        
        print("="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Duplicate rows
        duplicates = self.data.duplicated().sum()
        print(f"Duplicate Rows: {duplicates}")
        
        # Outliers detection for numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['ID']]
        
        print(f"\nAnalyzing outliers for {len(numerical_cols)} numerical columns...")
        outlier_summary = []
        
        for col in numerical_cols:
            try:
                valid_data = self.data[col].dropna()
                if len(valid_data) < len(self.data) * 0.1:
                    continue
                
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    outliers = ((valid_data < (Q1 - 1.5 * IQR)) | 
                               (valid_data > (Q3 + 1.5 * IQR))).sum()
                    if outliers > 0:
                        percentage = (outliers / len(self.data)) * 100
                        outlier_summary.append((col, outliers, percentage))
                        
            except Exception:
                continue
        
        # Show top 10 columns with most outliers
        outlier_summary.sort(key=lambda x: x[1], reverse=True)
        print(f"Top 10 Columns with Outliers:")
        for col, count, percentage in outlier_summary[:10]:
            print(f"  {col}: {count} outliers ({percentage:.2f}%)")
        
        # Data consistency checks
        print(f"\nData Consistency Checks:")
        
        # Age analysis
        if 'Age_Days' in self.data.columns:
            try:
                age_numeric = pd.to_numeric(self.data['Age_Days'], errors='coerce')
                age_years = abs(age_numeric.fillna(0)) / 365.25
                unrealistic_age = ((age_years < 18) | (age_years > 100)).sum()
                missing_age = age_numeric.isna().sum()
                print(f"  Age: {unrealistic_age} unrealistic, {missing_age} missing")
            except Exception:
                print(f"  Age: Could not analyze")
        
        # Income analysis
        if 'Client_Income' in self.data.columns:
            try:
                income_numeric = pd.to_numeric(self.data['Client_Income'], errors='coerce')
                negative_income = (income_numeric <= 0).sum()
                missing_income = income_numeric.isna().sum()
                print(f"  Income: {negative_income} non-positive, {missing_income} missing")
            except Exception:
                print(f"  Income: Could not analyze")
        
        print(f"✓ Data quality assessment completed!")
    
    def feature_engineering(self):
        """Create new features systematically"""
        if self.data is None:
            print("Please load data first")
            return
        
        print("="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Create a copy for processing
        self.processed_data = self.data.copy()
        original_features = self.processed_data.shape[1]
        
        # Ensure key columns are numeric
        key_numeric_cols = ['Age_Days', 'Employed_Days', 'Client_Income', 'Credit_Amount', 'Loan_Annuity']
        for col in key_numeric_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')
        
        new_features_created = []
        
        # 1. Age-related features
        if 'Age_Days' in self.processed_data.columns:
            self.processed_data['Age_Years'] = abs(self.processed_data['Age_Days'].fillna(0)) / 365.25
            self.processed_data['Age_Group'] = pd.cut(
                self.processed_data['Age_Years'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=['Young', 'Adult', 'Middle', 'Senior', 'Elder'],
                include_lowest=True
            )
            new_features_created.extend(['Age_Years', 'Age_Group'])
        
        # 2. Employment features
        if 'Employed_Days' in self.processed_data.columns:
            self.processed_data['Employment_Years'] = abs(self.processed_data['Employed_Days'].fillna(0)) / 365.25
            self.processed_data['Employment_Stability'] = pd.cut(
                self.processed_data['Employment_Years'],
                bins=[0, 1, 5, 10, 50],
                labels=['New', 'Stable', 'Experienced', 'Veteran'],
                include_lowest=True
            )
            new_features_created.extend(['Employment_Years', 'Employment_Stability'])
        
        # 3. Financial ratios
        if all(col in self.processed_data.columns for col in ['Client_Income', 'Credit_Amount']):
            self.processed_data['Income_Credit_Ratio'] = (
                self.processed_data['Client_Income'] / (self.processed_data['Credit_Amount'] + 1)
            )
            self.processed_data['Credit_Income_Percentage'] = (
                (self.processed_data['Credit_Amount'] / (self.processed_data['Client_Income'] + 1)) * 100
            )
            new_features_created.extend(['Income_Credit_Ratio', 'Credit_Income_Percentage'])
        
        if all(col in self.processed_data.columns for col in ['Client_Income', 'Loan_Annuity']):
            self.processed_data['Annuity_Income_Ratio'] = (
                self.processed_data['Loan_Annuity'] / (self.processed_data['Client_Income'] + 1)
            )
            new_features_created.append('Annuity_Income_Ratio')
        
        # 4. Asset ownership score
        asset_cols = ['Car_Owned', 'Bike_Owned', 'House_Own']
        available_assets = [col for col in asset_cols if col in self.processed_data.columns]
        if available_assets:
            self.processed_data['Asset_Score'] = self.processed_data[available_assets].sum(axis=1, skipna=True)
            new_features_created.append('Asset_Score')
        
        # 5. Contact completeness
        contact_cols = ['Mobile_Tag', 'Homephone_Tag', 'Workphone_Working']
        available_contacts = [col for col in contact_cols if col in self.processed_data.columns]
        if available_contacts:
            self.processed_data['Contact_Completeness'] = self.processed_data[available_contacts].sum(axis=1, skipna=True)
            new_features_created.append('Contact_Completeness')
        
        # 6. External scores
        score_cols = [col for col in self.processed_data.columns if 'Score_Source' in col]
        if len(score_cols) >= 2:
            self.processed_data['Average_External_Score'] = self.processed_data[score_cols].mean(axis=1, skipna=True)
            self.processed_data['Max_External_Score'] = self.processed_data[score_cols].max(axis=1, skipna=True)
            self.processed_data['Min_External_Score'] = self.processed_data[score_cols].min(axis=1, skipna=True)
            new_features_created.extend(['Average_External_Score', 'Max_External_Score', 'Min_External_Score'])
        
        # 7. Temporal features
        if 'Application_Process_Day' in self.processed_data.columns:
            self.processed_data['Is_Weekend'] = self.processed_data['Application_Process_Day'].isin([0, 6]).astype(int)
            new_features_created.append('Is_Weekend')
        
        # 8. Financial stress indicator
        if all(col in self.processed_data.columns for col in ['Credit_Income_Percentage', 'Active_Loan']):
            stress_score = (self.processed_data['Credit_Income_Percentage'] > 50).astype(int)
            if 'Active_Loan' in self.processed_data.columns:
                stress_score += self.processed_data['Active_Loan'].fillna(0)
            if 'Child_Count' in self.processed_data.columns:
                stress_score += (self.processed_data['Child_Count'] > 2).astype(int)
            self.processed_data['Financial_Stress_Score'] = stress_score
            new_features_created.append('Financial_Stress_Score')
        
        new_features_count = len(new_features_created)
        total_features = self.processed_data.shape[1]
        
        print(f"✓ Feature Engineering Completed!")
        print(f"  Original features: {original_features}")
        print(f"  New features created: {new_features_count}")
        print(f"  Total features: {total_features}")
        
        if new_features_created:
            print(f"\nNew features created:")
            for i, feature in enumerate(new_features_created, 1):
                print(f"  {i:2d}. {feature}")
        
        return self.processed_data
    
    def quick_eda(self):
        """Simplified EDA focusing on key insights"""
        if self.processed_data is None:
            print("Please run feature engineering first")
            return
        
        print("="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Generate statistical summary only
        self.generate_statistical_summary()
        
        # Create basic visualizations
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # 1. Target distribution
            if 'Default' in self.processed_data.columns:
                self.processed_data['Default'].value_counts().plot(kind='bar', ax=axes[0], color=['lightgreen', 'salmon'])
                axes[0].set_title('Target Variable Distribution')
                axes[0].set_xlabel('Default (0=No, 1=Yes)')
            
            # 2. Age distribution
            if 'Age_Years' in self.processed_data.columns:
                self.processed_data['Age_Years'].dropna().hist(bins=30, ax=axes[1], alpha=0.7, color='skyblue')
                axes[1].set_title('Age Distribution')
                axes[1].set_xlabel('Age (Years)')
            
            # 3. Income distribution
            if 'Client_Income' in self.processed_data.columns:
                income_data = self.processed_data['Client_Income'].dropna()
                income_data[income_data > 0].hist(bins=50, ax=axes[2], alpha=0.7, color='lightcoral')
                axes[2].set_title('Income Distribution')
                axes[2].set_xlabel('Income ($)')
            
            # 4. Asset Score vs Default
            if all(col in self.processed_data.columns for col in ['Asset_Score', 'Default']):
                asset_default = pd.crosstab(self.processed_data['Asset_Score'], self.processed_data['Default'], normalize='index') * 100
                asset_default.plot(kind='bar', ax=axes[3], color=['lightgreen', 'salmon'])
                axes[3].set_title('Default Rate by Asset Score')
                axes[3].legend(['No Default', 'Default'])
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def generate_statistical_summary(self):
        """Generate statistical summary"""
        print(f"Statistical Summary:")
        
        if 'Default' not in self.processed_data.columns:
            print("No target variable found")
            return
        
        # Top correlated features
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['ID', 'Default']]
        
        print(f"Top 10 Features Correlated with Default:")
        correlations = {}
        for col in numeric_cols:
            try:
                corr = self.processed_data[col].corr(self.processed_data['Default'])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                pass
        
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, corr) in enumerate(sorted_corr, 1):
            print(f"  {i:2d}. {feature:<30}: {corr:.4f}")
    
    def data_preprocessing_for_ml(self):
        """Prepare data for machine learning"""
        if self.processed_data is None:
            print("Please run feature engineering first")
            return None, None
        
        print("="*60)
        print("DATA PREPROCESSING FOR ML")
        print("="*60)
        
        ml_data = self.processed_data.copy()
        
        # Handle missing values
        print("Handling missing values...")
        numeric_cols = ml_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if ml_data[col].isnull().sum() > 0:
                ml_data[col].fillna(ml_data[col].median(), inplace=True)
        
        categorical_cols = ml_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if ml_data[col].isnull().sum() > 0:
                try:
                    ml_data[col].fillna(ml_data[col].mode()[0], inplace=True)
                except:
                    ml_data[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        for col in categorical_cols:
            if col != 'Default':
                le = LabelEncoder()
                try:
                    ml_data[col] = le.fit_transform(ml_data[col].astype(str))
                except:
                    print(f"  Warning: Could not encode {col}")
        
        # Remove ID if present
        if 'ID' in ml_data.columns:
            ml_data.drop('ID', axis=1, inplace=True)
        
        # Separate features and target
        if 'Default' in ml_data.columns:
            X = ml_data.drop('Default', axis=1)
            y = ml_data['Default']
        else:
            X = ml_data
            y = None
        
        print(f"✓ Final dataset prepared:")
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape if y is not None else 'None'}")
        
        return X, y
    
    def run_complete_pipeline(self, file_path):
        """Run the complete pipeline"""
        print("🚀 STARTING LOAN DEFAULT PREDICTION PIPELINE")
        print("="*80)
        
        try:
            # Step 1: Load data
            if self.load_data(file_path) is None:
                return None, None
            
            # Step 2: Initial exploration
            self.initial_data_exploration()
            
            # Step 3: Data quality check
            self.data_quality_check()
            
            # Step 4: Feature engineering
            self.feature_engineering()
            
            # Step 5: Quick EDA
            self.quick_eda()
            
            # Step 6: Preprocessing
            X, y = self.data_preprocessing_for_ml()
            
            if X is not None and y is not None:
                print("\n" + "="*60)
                print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"✓ Final dataset: {X.shape[0]} rows, {X.shape[1]} features")
                print(f"✓ Target distribution: {y.value_counts().to_dict()}")
                print(f"✓ Class imbalance ratio: {(y==0).sum()/(y==1).sum():.2f}:1")
                
                # Save processed data
                final_data = X.copy()
                final_data['Default'] = y
                final_data.to_csv('processed_loan_data.csv', index=False)
                print(f"✓ Processed data saved to 'processed_loan_data.csv'")
                
                print(f"\n🚀 READY FOR NEXT STEPS:")
                print(f"   1. ✅ Data preprocessing completed")
                print(f"   2. 🎯 Apply SMOTE for class imbalance")
                print(f"   3. 🤖 Train multiple ML models")
                print(f"   4. 📊 Evaluate and compare models")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    """Main execution function"""
    print("="*80)
    print("LOAN DEFAULT PREDICTION - STABLE PIPELINE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = LoanDefaultMLPipeline()
    
    # Run pipeline
    X, y = pipeline.run_complete_pipeline('Dataset.csv')
    
    if X is not None and y is not None:
        print(f"\n✅ SUCCESS! Pipeline completed without errors.")
        print(f"📊 Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Target variable: {y.shape[0]} samples")
        print(f"⚖️  Imbalance ratio: {(y==0).sum()/(y==1).sum():.2f}:1")
        
        return X, y
    else:
        print(f"\n❌ Pipeline failed - check errors above")
        return None, None


if __name__ == "__main__":
    # Set up environment to avoid PyArrow issues
    try:
        X, y = main()
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()