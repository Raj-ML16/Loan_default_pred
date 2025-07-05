# Standalone FastAPI with Integrated Monitoring
# Filename: fastapi_standalone.py
# Purpose: Complete FastAPI app with monitoring - no external dependencies

import pandas as pd
import numpy as np
import joblib
import json
import sqlite3
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("⚠️ FastAPI not installed. Run: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

# ML imports
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Pydantic models for your specific loan application
class LoanApplication(BaseModel):
    """Loan application with your actual 20 selected features"""
    
    # Customer ID
    customer_id: Optional[str] = Field(None, description="Unique customer identifier")
    
    # Your actual 20 selected features
    Credit_Amount: float = Field(..., ge=0, le=1000000, description="Credit amount requested")
    Client_Income_Type: int = Field(..., ge=0, le=10, description="Client income type category")
    Client_Education: int = Field(..., ge=0, le=10, description="Client education level")
    Client_Gender: int = Field(..., ge=0, le=2, description="Client gender (0=Female, 1=Male, 2=Other)")
    Age_Days: int = Field(..., ge=0, le=50000, description="Age in days")
    Employed_Days: int = Field(..., ge=0, le=20000, description="Days employed")
    Registration_Days: int = Field(..., ge=0, le=10000, description="Days since registration")
    ID_Days: int = Field(..., ge=0, le=10000, description="Days since ID issued")
    Cleint_City_Rating: int = Field(..., ge=1, le=5, description="Client city rating")
    Client_Permanent_Match_Tag: int = Field(..., ge=0, le=1, description="Permanent address match (0=No, 1=Yes)")
    Type_Organization: int = Field(..., ge=0, le=20, description="Organization type category")
    Score_Source_1: float = Field(..., ge=0, le=1, description="External score source 1")
    Score_Source_2: float = Field(..., ge=0, le=1, description="External score source 2")
    Score_Source_3: float = Field(..., ge=0, le=1, description="External score source 3")
    Phone_Change: int = Field(..., ge=0, le=10, description="Number of phone changes")
    Age_Years: int = Field(..., ge=18, le=100, description="Age in years")
    Employment_Years: int = Field(..., ge=0, le=50, description="Years of employment")
    Average_External_Score: float = Field(..., ge=0, le=1, description="Average external credit score")
    Max_External_Score: float = Field(..., ge=0, le=1, description="Maximum external credit score")
    Min_External_Score: float = Field(..., ge=0, le=1, description="Minimum external credit score")
    
    @validator('customer_id', pre=True, always=True)
    def set_customer_id(cls, v):
        return v or f"customer_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}"
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "sample_customer_001",
                "Credit_Amount": 15000,
                "Client_Income_Type": 2,
                "Client_Education": 3,
                "Client_Gender": 1,
                "Age_Days": 12775,  # ~35 years
                "Employed_Days": 1095,  # ~3 years
                "Registration_Days": 365,
                "ID_Days": 1825,  # ~5 years
                "Cleint_City_Rating": 3,
                "Client_Permanent_Match_Tag": 1,
                "Type_Organization": 5,
                "Score_Source_1": 0.65,
                "Score_Source_2": 0.70,
                "Score_Source_3": 0.60,
                "Phone_Change": 1,
                "Age_Years": 35,
                "Employment_Years": 3,
                "Average_External_Score": 0.65,
                "Max_External_Score": 0.75,
                "Min_External_Score": 0.55
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response model"""
    customer_id: str
    probability: float = Field(..., ge=0, le=1, description="Default probability")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0=No Default, 1=Default)")
    decision: str = Field(..., description="Business decision (APPROVE/REJECT/REVIEW)")
    confidence: str = Field(..., description="Confidence level (HIGH/MEDIUM/LOW)")
    threshold_used: float = Field(..., description="Decision threshold used")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

class FeedbackRequest(BaseModel):
    """Feedback request model"""
    customer_id: str = Field(..., description="Customer ID")
    actual_outcome: int = Field(..., ge=0, le=1, description="Actual outcome (0=No Default, 1=Default)")

class MLModelService:
    """ML Model service for your specific model"""
    
    def __init__(self, model_path: str, log_db_path: str = 'model_predictions.db'):
        self.model_path = model_path
        self.log_db_path = log_db_path
        self.model_package = None
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.optimal_threshold = 0.5
        self.original_features = None
        self.selected_features = [
            'Credit_Amount', 'Client_Income_Type', 'Client_Education', 'Client_Gender',
            'Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days',
            'Cleint_City_Rating', 'Client_Permanent_Match_Tag', 'Type_Organization',
            'Score_Source_1', 'Score_Source_2', 'Score_Source_3', 'Phone_Change',
            'Age_Years', 'Employment_Years', 'Average_External_Score',
            'Max_External_Score', 'Min_External_Score'
        ]
        self.start_time = datetime.now()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize
        self.load_model()
        self.setup_database()
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model_package = joblib.load(self.model_path)
            self.model = self.model_package['model']
            self.scaler = self.model_package.get('scaler')
            self.feature_selector = self.model_package.get('feature_selector')
            
            # Get original features from scaler
            if self.scaler and hasattr(self.scaler, 'feature_names_in_'):
                self.original_features = list(self.scaler.feature_names_in_)
            
            # Get optimal threshold
            if 'optimization_config' in self.model_package:
                self.optimal_threshold = self.model_package['optimization_config'].get('optimal_threshold', 0.5)
            
            print(f"✅ Model loaded successfully!")
            print(f"   📊 Original features: {len(self.original_features) if self.original_features else 'Unknown'}")
            print(f"   🎯 Selected features: {len(self.selected_features)}")
            print(f"   🎚️ Optimal threshold: {self.optimal_threshold}")
            
            # Show model metrics
            if 'optimized_metrics' in self.model_package:
                metrics = self.model_package['optimized_metrics']
                precision = metrics.get('test_precision', 0)
                print(f"   📈 Model precision: {precision:.1%}")
            
            self.logger.info(f"Model loaded: {len(self.selected_features)} features, threshold: {self.optimal_threshold}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            print(f"❌ Model loading failed: {e}")
            return False
    
    def setup_database(self):
        """Setup SQLite database"""
        try:
            conn = sqlite3.connect(self.log_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    customer_id TEXT,
                    features TEXT NOT NULL,
                    probability REAL NOT NULL,
                    prediction INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    confidence_level TEXT NOT NULL,
                    processing_time_ms REAL,
                    actual_outcome INTEGER,
                    feedback_date TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_predictions INTEGER,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    approval_rate REAL,
                    avg_confidence REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Database setup completed")
            print("✅ Database setup completed")
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            print(f"❌ Database setup failed: {e}")
    
    def preprocess_features(self, loan_data: LoanApplication) -> np.ndarray:
        """Preprocess loan application data through the exact pipeline"""
        try:
            # Get the input data as dict (excluding customer_id)
            input_data = loan_data.dict(exclude={'customer_id'})
            
            # Create full original feature vector with zeros
            if self.original_features:
                full_data = {name: 0.0 for name in self.original_features}
                
                # Map our 20 selected features to the original 52 features
                for feature_name, value in input_data.items():
                    if feature_name in self.original_features:
                        full_data[feature_name] = value
                
                # Create DataFrame with all 52 original features in correct order
                df = pd.DataFrame([full_data])
                df = df[self.original_features]  # Ensure correct order
                
                # Apply scaler (transforms 52 -> 52)
                if self.scaler:
                    scaled_features = self.scaler.transform(df)
                else:
                    scaled_features = df.values
                
                # Apply feature selector (transforms 52 -> 20)
                if self.feature_selector:
                    selected_features = self.feature_selector.transform(scaled_features)
                else:
                    selected_features = scaled_features
            else:
                # Fallback: direct feature mapping
                df = pd.DataFrame([input_data])
                df = df[self.selected_features]  # Ensure correct order
                selected_features = df.values
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise HTTPException(status_code=422, detail=f"Feature preprocessing failed: {e}")
    
    async def predict(self, loan_data: LoanApplication) -> PredictionResponse:
        """Make prediction"""
        start_time = datetime.now()
        
        try:
            # Preprocess features
            processed_features = self.preprocess_features(loan_data)
            
            # Get prediction probability
            probability = float(self.model.predict_proba(processed_features)[0, 1])
            
            # Apply optimal threshold
            binary_prediction = int(probability >= self.optimal_threshold)
            
            # Business decision logic (optimized for 37% precision)
            if probability <= 0.15:
                decision = "APPROVE"
                confidence = "HIGH"
            elif probability <= 0.25:
                decision = "APPROVE"
                confidence = "MEDIUM"
            elif probability <= 0.40:
                decision = "REVIEW"
                confidence = "MEDIUM"
            else:
                decision = "REJECT"
                confidence = "HIGH"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create response
            response = PredictionResponse(
                customer_id=loan_data.customer_id,
                probability=probability,
                prediction=binary_prediction,
                decision=decision,
                confidence=confidence,
                threshold_used=self.optimal_threshold,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time
            )
            
            # Log prediction in background
            await self.log_prediction_async(response, loan_data, processing_time)
            
            self.logger.info(f"Prediction for {loan_data.customer_id}: {decision} ({probability:.4f})")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    async def log_prediction_async(self, response: PredictionResponse, 
                                  loan_data: LoanApplication, processing_time: float):
        """Log prediction asynchronously"""
        try:
            conn = sqlite3.connect(self.log_db_path)
            cursor = conn.cursor()
            
            features_json = json.dumps(loan_data.dict(exclude={'customer_id'}))
            
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, customer_id, features, probability, prediction, 
                 decision, confidence_level, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                response.timestamp,
                response.customer_id,
                features_json,
                response.probability,
                response.prediction,
                response.decision,
                response.confidence,
                processing_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log prediction: {e}")
    
    def get_stats(self) -> dict:
        """Get prediction statistics"""
        try:
            conn = sqlite3.connect(self.log_db_path)
            
            stats_query = '''
                SELECT 
                    COUNT(*) as total_predictions,
                    COALESCE(AVG(probability), 0) as avg_probability,
                    COUNT(CASE WHEN decision = 'APPROVE' THEN 1 END) as approvals,
                    COUNT(CASE WHEN decision = 'REJECT' THEN 1 END) as rejections,
                    COUNT(CASE WHEN decision = 'REVIEW' THEN 1 END) as reviews,
                    COALESCE(AVG(processing_time_ms), 0) as avg_processing_time_ms
                FROM predictions
            '''
            
            result = pd.read_sql_query(stats_query, conn).iloc[0]
            conn.close()
            
            return {
                "total_predictions": int(result['total_predictions']),
                "avg_probability": float(result['avg_probability']),
                "approvals": int(result['approvals']),
                "rejections": int(result['rejections']),
                "reviews": int(result['reviews']),
                "approval_rate": float(result['approvals'] / result['total_predictions']) if result['total_predictions'] > 0 else 0.0,
                "avg_processing_time_ms": float(result['avg_processing_time_ms'])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")

def create_app(model_path: str) -> FastAPI:
    """Create FastAPI application"""
    
    # Initialize model service
    print("\n🔧 Initializing model service...")
    model_service = MLModelService(model_path)
    
    if not model_service.model:
        raise Exception("Failed to load model")
    
    app = FastAPI(
        title="Loan Default Prediction API",
        description="Production loan default prediction API with 37.2% precision using 20 optimized features",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        uptime = (datetime.now() - model_service.start_time).total_seconds()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model_service.model is not None,
            "model_name": model_service.model_package.get('model_name', 'Unknown'),
            "model_precision": f"{model_service.model_package.get('optimized_metrics', {}).get('test_precision', 0):.1%}",
            "feature_count": len(model_service.selected_features),
            "uptime_seconds": uptime
        }
    
    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict_default(loan_data: LoanApplication):
        """Make loan default prediction using 20 optimized features"""
        return await model_service.predict(loan_data)
    
    @app.get("/stats", tags=["Statistics"])
    async def get_statistics():
        """Get prediction statistics"""
        return model_service.get_stats()
    
    @app.post("/feedback", tags=["Feedback"])
    async def update_feedback(feedback: FeedbackRequest):
        """Update actual loan outcome for monitoring"""
        try:
            conn = sqlite3.connect(model_service.log_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE predictions 
                SET actual_outcome = ?, feedback_date = ?
                WHERE customer_id = ? AND actual_outcome IS NULL
            ''', (feedback.actual_outcome, datetime.now().isoformat(), feedback.customer_id))
            
            rows_affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            if rows_affected > 0:
                return {"status": "success", "message": "Feedback updated successfully"}
            else:
                raise HTTPException(status_code=404, detail="Customer ID not found or already has feedback")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update feedback: {e}")
    
    @app.get("/features", tags=["Model"])
    async def get_feature_info():
        """Get information about the 20 required features"""
        return {
            "feature_count": len(model_service.selected_features),
            "selected_features": model_service.selected_features,
            "model_info": {
                "precision": f"{model_service.model_package.get('optimized_metrics', {}).get('test_precision', 0):.1%}",
                "threshold": model_service.optimal_threshold,
                "total_original_features": len(model_service.original_features) if model_service.original_features else None
            }
        }
    
    return app

def main():
    """Main function"""
    print("🚀 STANDALONE FASTAPI PRODUCTION API")
    print("="*50)
    
    model_path = "fast_optimized_model_precision_optimized_20250630_233735.pkl"
    
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available")
        return None
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("📁 Available .pkl files:")
        for file in Path('.').glob('*.pkl'):
            print(f"   - {file}")
        return None
    
    try:
        app = create_app(model_path)
        print("\n✅ FastAPI application created successfully!")
        print("\n🎯 Available Endpoints:")
        print("   GET  /health      - Health check")
        print("   POST /predict     - Make predictions")
        print("   GET  /stats       - Prediction statistics")
        print("   POST /feedback    - Update actual outcomes")
        print("   GET  /features    - Feature information")
        print("   GET  /docs        - Interactive API docs")
        
        return app
        
    except Exception as e:
        print(f"❌ Failed to create FastAPI app: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    app = main()
    
    if app:
        print("\n🚀 Starting FastAPI server...")
        print("🌐 API: http://localhost:8000")
        print("📚 Docs: http://localhost:8000/docs")
        print("🛑 Press Ctrl+C to stop")
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
    else:
        print("❌ Failed to initialize FastAPI application")