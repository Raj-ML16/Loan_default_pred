# FastAPI with Integrated Phase 5 Monitoring
# Filename: fastapi_with_monitoring.py
# Purpose: Your existing FastAPI enhanced with Phase 5 monitoring

import sys
import os
from pathlib import Path

# Import your existing FastAPI code
from fastapi_working_final import create_fastapi_app, MLModelService, LoanApplication
from monitoring_system import MonitoringOrchestrator, create_monitoring_endpoints

# Additional imports for enhanced monitoring
import uvicorn
import asyncio
from datetime import datetime
import threading
import time

class EnhancedProductionAPI:
    """
    Enhanced production API with integrated monitoring
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.app = None
        self.monitoring_orchestrator = None
        self.monitoring_thread = None
        
        print("🚀 Enhanced Production API with Phase 5 Monitoring")
        print("="*60)
    
    def initialize_app(self):
        """Initialize FastAPI app with monitoring"""
        try:
            # Create base FastAPI app
            print("🔧 Creating base FastAPI application...")
            self.app = create_fastapi_app(self.model_path)
            
            # Initialize monitoring system
            print("📊 Initializing monitoring system...")
            self.monitoring_orchestrator = MonitoringOrchestrator()
            
            # Add monitoring endpoints
            print("🔗 Adding monitoring endpoints...")
            create_monitoring_endpoints(self.app, self.monitoring_orchestrator)
            
            # Add custom monitoring endpoints
            self._add_custom_endpoints()
            
            print("✅ Enhanced API initialized successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ API initialization failed: {e}")
            return False
    
    def _add_custom_endpoints(self):
        """Add custom monitoring endpoints"""
        
        @self.app.get("/admin/monitoring-summary", tags=["Admin"])
        async def get_monitoring_summary():
            """Get comprehensive monitoring summary"""
            try:
                # Get all metrics
                health = self.monitoring_orchestrator.run_health_check()
                metrics = self.monitoring_orchestrator.performance_monitor.calculate_performance_metrics(7)
                roi = self.monitoring_orchestrator.bi_reporter.calculate_roi_metrics(30)
                
                summary = {
                    "timestamp": datetime.now().isoformat(),
                    "system_health": health.get('overall_status', 'UNKNOWN'),
                    "active_alerts": len(health.get('alerts', [])),
                    "last_7_days": {
                        "total_predictions": metrics.get('total_predictions', 0),
                        "approval_rate": metrics.get('approval_rate', 0),
                        "avg_probability": metrics.get('avg_probability', 0),
                        "avg_processing_time_ms": metrics.get('avg_processing_time_ms', 0)
                    },
                    "last_30_days_roi": {
                        "total_predictions": roi.get('total_predictions', 0),
                        "automation_rate": roi.get('efficiency_metrics', {}).get('automation_rate', 0),
                        "net_savings": roi.get('savings', {}).get('net_savings', 0)
                    },
                    "model_performance": metrics.get('performance_metrics', {}),
                    "recommendations": health.get('recommendations', [])
                }
                
                return summary
                
            except Exception as e:
                return {"error": f"Failed to generate summary: {e}"}
        
        @self.app.get("/admin/model-status", tags=["Admin"])
        async def get_model_status():
            """Get detailed model status"""
            try:
                # Check recent performance
                metrics = self.monitoring_orchestrator.performance_monitor.calculate_performance_metrics(7)
                drift = self.monitoring_orchestrator.performance_monitor.detect_data_drift()
                
                # Model health assessment
                health_score = 100  # Start with perfect score
                warnings = []
                
                # Check performance metrics
                if metrics.get('performance_metrics'):
                    perf = metrics['performance_metrics']
                    if perf.get('precision', 1) < 0.35:
                        health_score -= 30
                        warnings.append("Precision below target (35%)")
                    if perf.get('f1_score', 1) < 0.20:
                        health_score -= 20
                        warnings.append("F1-score below minimum (20%)")
                
                # Check processing time
                if metrics.get('avg_processing_time_ms', 0) > 200:
                    health_score -= 10
                    warnings.append("Slow response times")
                
                # Check data drift
                drift_detected = False
                if 'drift_results' in drift:
                    for key, result in drift['drift_results'].items():
                        if result.get('drift_detected', False):
                            drift_detected = True
                            health_score -= 15
                            warnings.append(f"Data drift detected in {key}")
                
                # Determine status
                if health_score >= 90:
                    status = "EXCELLENT"
                elif health_score >= 75:
                    status = "GOOD"
                elif health_score >= 60:
                    status = "WARNING"
                else:
                    status = "CRITICAL"
                
                return {
                    "model_status": status,
                    "health_score": health_score,
                    "warnings": warnings,
                    "last_updated": datetime.now().isoformat(),
                    "uptime_info": {
                        "predictions_last_7_days": metrics.get('total_predictions', 0),
                        "avg_daily_volume": metrics.get('total_predictions', 0) / 7,
                        "data_drift_detected": drift_detected
                    },
                    "performance_summary": metrics.get('performance_metrics', {})
                }
                
            except Exception as e:
                return {"error": f"Failed to get model status: {e}"}
        
        @self.app.post("/admin/trigger-monitoring", tags=["Admin"])
        async def trigger_monitoring():
            """Manually trigger monitoring cycle"""
            try:
                result = self.monitoring_orchestrator.run_monitoring_cycle()
                return {
                    "status": "success",
                    "message": "Monitoring cycle completed",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"Monitoring cycle failed: {e}"}
        
        @self.app.get("/admin/generate-report", tags=["Admin"])
        async def generate_business_report():
            """Generate business intelligence report"""
            try:
                report = self.monitoring_orchestrator.generate_daily_report()
                return {
                    "status": "success",
                    "report_generated": True,
                    "report_content": report,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": f"Report generation failed: {e}"}
    
    def start_background_monitoring(self, interval_minutes: int = 60):
        """Start background monitoring thread"""
        def monitoring_worker():
            """Background monitoring worker"""
            print(f"🔄 Background monitoring started (every {interval_minutes} minutes)")
            
            while True:
                try:
                    print(f"\n⏰ Running scheduled monitoring cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Run health check
                    health = self.monitoring_orchestrator.run_health_check()
                    
                    # Log status
                    status = health.get('overall_status', 'UNKNOWN')
                    alerts = len(health.get('alerts', []))
                    
                    print(f"✅ Health check completed - Status: {status}, Alerts: {alerts}")
                    
                    # Generate daily report if it's morning
                    current_hour = datetime.now().hour
                    if current_hour == 8:  # 8 AM
                        print("📋 Generating daily report...")
                        self.monitoring_orchestrator.generate_daily_report()
                    
                    # Update dashboard
                    if alerts == 0:  # Only update if no critical issues
                        self.monitoring_orchestrator.create_live_dashboard()
                    
                except Exception as e:
                    print(f"❌ Background monitoring error: {e}")
                
                # Wait for next cycle
                time.sleep(interval_minutes * 60)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        print(f"✅ Background monitoring started (every {interval_minutes} minutes)")
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8000, 
                   enable_background_monitoring: bool = True):
        """Run the enhanced server"""
        if not self.app:
            print("❌ App not initialized. Call initialize_app() first.")
            return
        
        # Start background monitoring
        if enable_background_monitoring:
            self.start_background_monitoring(interval_minutes=30)  # Every 30 minutes
        
        # Run initial monitoring cycle
        print("\n🔄 Running initial monitoring cycle...")
        initial_result = self.monitoring_orchestrator.run_monitoring_cycle()
        
        if initial_result['status'] == 'SUCCESS':
            print("✅ Initial monitoring completed successfully")
        else:
            print(f"⚠️ Initial monitoring had issues: {initial_result.get('error', 'Unknown')}")
        
        # Display available endpoints
        print(f"\n🚀 Enhanced API Server Starting...")
        print(f"🌐 Main API: http://{host}:{port}")
        print(f"📚 API Docs: http://{host}:{port}/docs")
        print(f"📊 Live Dashboard: http://{host}:{port}/monitoring/dashboard")
        print(f"🏥 Health Check: http://{host}:{port}/monitoring/health")
        print(f"📈 ROI Metrics: http://{host}:{port}/monitoring/roi")
        print(f"👤 Admin Summary: http://{host}:{port}/admin/monitoring-summary")
        
        print(f"\n🔧 Key Features Enabled:")
        print(f"   ✅ 37.2% Precision Model Deployment")
        print(f"   ✅ Real-time Performance Monitoring")
        print(f"   ✅ Automated Alerting System")
        print(f"   ✅ Business Intelligence Reporting")
        print(f"   ✅ Data Drift Detection")
        print(f"   ✅ Interactive Dashboards")
        print(f"   ✅ Background Health Monitoring")
        
        print(f"\n🛑 Press Ctrl+C to stop the server")
        
        # Start the server
        try:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")


def main():
    """Main function to run enhanced production API"""
    print("🎯 PHASE 5: PRODUCTION API WITH ADVANCED MONITORING")
    print("="*70)
    
    # Configuration
    model_path = "fast_optimized_model_precision_optimized_20250630_233735.pkl"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("📁 Available .pkl files:")
        for file in Path('.').glob('*.pkl'):
            print(f"   - {file}")
        return
    
    # Initialize enhanced API
    try:
        enhanced_api = EnhancedProductionAPI(model_path)
        
        if enhanced_api.initialize_app():
            # Run the server
            enhanced_api.run_server(
                host="0.0.0.0",
                port=8000,
                enable_background_monitoring=True
            )
        else:
            print("❌ Failed to initialize enhanced API")
            
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()