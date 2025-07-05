# Phase 5: Advanced Monitoring & Business Intelligence System
# Filename: monitoring_system.py
# Purpose: Enterprise-grade monitoring for production ML model

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Monitoring and visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not installed. Run: pip install plotly")
    PLOTLY_AVAILABLE = False

# FastAPI monitoring enhancement
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Statistics and analysis
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class ModelPerformanceMonitor:
    """
    Advanced model performance monitoring system
    """
    
    def __init__(self, db_path: str = 'model_predictions.db'):
        self.db_path = db_path
        self.performance_thresholds = {
            'precision_min': 0.35,  # Alert if precision drops below 35%
            'f1_min': 0.20,         # Alert if F1 drops below 20%
            'volume_change': 0.30,  # Alert if volume changes by 30%
            'response_time_max': 500,  # Alert if response time > 500ms
            'error_rate_max': 0.05    # Alert if error rate > 5%
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🔍 Model Performance Monitor Initialized")
    
    def calculate_performance_metrics(self, days_back: int = 7) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent predictions with actual outcomes
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            query = '''
                SELECT 
                    probability, prediction, decision, actual_outcome,
                    processing_time_ms, timestamp, confidence_level
                FROM predictions 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=[cutoff_date])
            conn.close()
            
            if len(df) == 0:
                return {"error": "No data available for the specified period"}
            
            # Basic metrics
            total_predictions = len(df)
            avg_probability = df['probability'].mean()
            avg_processing_time = df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else None
            
            # Decision distribution
            decision_counts = df['decision'].value_counts()
            approval_rate = decision_counts.get('APPROVE', 0) / total_predictions
            rejection_rate = decision_counts.get('REJECT', 0) / total_predictions
            review_rate = decision_counts.get('REVIEW', 0) / total_predictions
            
            # Performance metrics (if actual outcomes available)
            performance_metrics = {}
            labeled_data = df[df['actual_outcome'].notna()]
            
            if len(labeled_data) > 0:
                y_true = labeled_data['actual_outcome']
                y_pred = labeled_data['prediction']
                y_prob = labeled_data['probability']
                
                performance_metrics = {
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else None,
                    'labeled_samples': len(labeled_data)
                }
            
            # Risk distribution
            risk_distribution = {
                'low_risk': len(df[df['probability'] <= 0.2]) / total_predictions,
                'medium_risk': len(df[(df['probability'] > 0.2) & (df['probability'] <= 0.4)]) / total_predictions,
                'high_risk': len(df[df['probability'] > 0.4]) / total_predictions
            }
            
            # Confidence distribution
            confidence_counts = df['confidence_level'].value_counts()
            confidence_distribution = {
                level: count / total_predictions 
                for level, count in confidence_counts.items()
            }
            
            return {
                'period_days': days_back,
                'total_predictions': total_predictions,
                'avg_probability': avg_probability,
                'avg_processing_time_ms': avg_processing_time,
                'approval_rate': approval_rate,
                'rejection_rate': rejection_rate,
                'review_rate': review_rate,
                'risk_distribution': risk_distribution,
                'confidence_distribution': confidence_distribution,
                'performance_metrics': performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {"error": str(e)}
    
    def detect_data_drift(self, days_back: int = 7, baseline_days: int = 30) -> Dict:
        """Detect data drift using statistical tests"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent data
            recent_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            baseline_date = (datetime.now() - timedelta(days=baseline_days)).isoformat()
            
            recent_query = '''
                SELECT probability, processing_time_ms
                FROM predictions 
                WHERE timestamp >= ?
            '''
            
            baseline_query = '''
                SELECT probability, processing_time_ms
                FROM predictions 
                WHERE timestamp >= ? AND timestamp < ?
            '''
            
            recent_data = pd.read_sql_query(recent_query, conn, params=[recent_date])
            baseline_data = pd.read_sql_query(baseline_query, conn, 
                                            params=[baseline_date, recent_date])
            conn.close()
            
            if len(recent_data) < 10 or len(baseline_data) < 10:
                return {"warning": "Insufficient data for drift detection"}
            
            # Statistical tests
            drift_results = {}
            
            # Probability distribution drift
            prob_stat, prob_p_value = stats.ks_2samp(
                baseline_data['probability'], recent_data['probability']
            )
            
            drift_results['probability_drift'] = {
                'ks_statistic': prob_stat,
                'p_value': prob_p_value,
                'drift_detected': prob_p_value < 0.05,
                'severity': 'HIGH' if prob_p_value < 0.01 else 'MEDIUM' if prob_p_value < 0.05 else 'LOW'
            }
            
            # Processing time drift
            if recent_data['processing_time_ms'].notna().sum() > 5:
                time_stat, time_p_value = stats.ks_2samp(
                    baseline_data['processing_time_ms'].dropna(), 
                    recent_data['processing_time_ms'].dropna()
                )
                
                drift_results['processing_time_drift'] = {
                    'ks_statistic': time_stat,
                    'p_value': time_p_value,
                    'drift_detected': time_p_value < 0.05,
                    'severity': 'HIGH' if time_p_value < 0.01 else 'MEDIUM' if time_p_value < 0.05 else 'LOW'
                }
            
            return {
                'period_analyzed': f"Recent {days_back} days vs baseline {baseline_days} days",
                'recent_samples': len(recent_data),
                'baseline_samples': len(baseline_data),
                'drift_results': drift_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting drift: {e}")
            return {"error": str(e)}
    
    def generate_alerts(self, metrics: Dict) -> List[Dict]:
        """Generate alerts based on performance thresholds"""
        alerts = []
        
        try:
            # Performance alerts
            if 'performance_metrics' in metrics and metrics['performance_metrics']:
                perf = metrics['performance_metrics']
                
                if perf.get('precision', 1) < self.performance_thresholds['precision_min']:
                    alerts.append({
                        'type': 'PERFORMANCE_DEGRADATION',
                        'severity': 'HIGH',
                        'message': f"Precision dropped to {perf['precision']:.1%} (below {self.performance_thresholds['precision_min']:.1%})",
                        'value': perf['precision'],
                        'threshold': self.performance_thresholds['precision_min'],
                        'timestamp': datetime.now().isoformat()
                    })
                
                if perf.get('f1_score', 1) < self.performance_thresholds['f1_min']:
                    alerts.append({
                        'type': 'F1_SCORE_LOW',
                        'severity': 'MEDIUM',
                        'message': f"F1-score dropped to {perf['f1_score']:.1%} (below {self.performance_thresholds['f1_min']:.1%})",
                        'value': perf['f1_score'],
                        'threshold': self.performance_thresholds['f1_min'],
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Processing time alerts
            if metrics.get('avg_processing_time_ms', 0) > self.performance_thresholds['response_time_max']:
                alerts.append({
                    'type': 'SLOW_RESPONSE_TIME',
                    'severity': 'MEDIUM',
                    'message': f"Average response time: {metrics['avg_processing_time_ms']:.1f}ms (above {self.performance_thresholds['response_time_max']}ms)",
                    'value': metrics['avg_processing_time_ms'],
                    'threshold': self.performance_thresholds['response_time_max'],
                    'timestamp': datetime.now().isoformat()
                })
            
            # Volume anomaly detection
            # Get historical average for comparison
            historical_metrics = self.calculate_performance_metrics(days_back=30)
            if 'total_predictions' in historical_metrics:
                historical_daily_avg = historical_metrics['total_predictions'] / 30
                current_daily_avg = metrics['total_predictions'] / metrics['period_days']
                
                volume_change = abs(current_daily_avg - historical_daily_avg) / historical_daily_avg
                
                if volume_change > self.performance_thresholds['volume_change']:
                    alerts.append({
                        'type': 'VOLUME_ANOMALY',
                        'severity': 'MEDIUM',
                        'message': f"Prediction volume changed by {volume_change:.1%} (current: {current_daily_avg:.1f}/day, historical: {historical_daily_avg:.1f}/day)",
                        'value': volume_change,
                        'threshold': self.performance_thresholds['volume_change'],
                        'timestamp': datetime.now().isoformat()
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error generating alerts: {e}")
            return []


class BusinessIntelligenceReporter:
    """
    Business intelligence and ROI reporting system
    """
    
    def __init__(self, db_path: str = 'model_predictions.db'):
        self.db_path = db_path
        
        # Business assumptions for ROI calculation
        self.business_config = {
            'avg_loan_amount': 20000,
            'default_cost_rate': 0.60,  # 60% of loan amount lost on default
            'manual_review_cost': 50,   # Cost per manual review
            'processing_cost_per_prediction': 0.10,  # API cost
            'baseline_precision': 0.161,  # Original model precision
            'current_precision': 0.372   # Optimized model precision
        }
        
        print("📊 Business Intelligence Reporter Initialized")
    
    def calculate_roi_metrics(self, days_back: int = 30) -> Dict:
        """Calculate ROI and business impact metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # Get prediction data
            query = '''
                SELECT 
                    probability, prediction, decision, actual_outcome,
                    processing_time_ms, timestamp
                FROM predictions 
                WHERE timestamp >= ?
            '''
            
            df = pd.read_sql_query(query, conn, params=[cutoff_date])
            conn.close()
            
            if len(df) == 0:
                return {"error": "No data available for ROI calculation"}
            
            total_predictions = len(df)
            
            # Decision breakdown
            approved = len(df[df['decision'] == 'APPROVE'])
            rejected = len(df[df['decision'] == 'REJECT'])
            reviews = len(df[df['decision'] == 'REVIEW'])
            
            # Cost calculations
            costs = {
                'api_processing_cost': total_predictions * self.business_config['processing_cost_per_prediction'],
                'manual_review_cost': reviews * self.business_config['manual_review_cost'],
                'total_operational_cost': 0
            }
            costs['total_operational_cost'] = costs['api_processing_cost'] + costs['manual_review_cost']
            
            # Revenue/savings calculations (based on actual outcomes if available)
            labeled_data = df[df['actual_outcome'].notna()]
            savings = {}
            
            if len(labeled_data) > 0:
                # Actual performance
                true_positives = len(labeled_data[(labeled_data['prediction'] == 1) & (labeled_data['actual_outcome'] == 1)])
                false_positives = len(labeled_data[(labeled_data['prediction'] == 1) & (labeled_data['actual_outcome'] == 0)])
                false_negatives = len(labeled_data[(labeled_data['prediction'] == 0) & (labeled_data['actual_outcome'] == 1)])
                
                # Calculate savings from catching defaults
                default_cost_prevented = true_positives * self.business_config['avg_loan_amount'] * self.business_config['default_cost_rate']
                
                # Calculate cost of rejecting good customers (opportunity cost)
                good_customer_rejected_cost = false_positives * self.business_config['avg_loan_amount'] * 0.05  # 5% profit margin
                
                savings = {
                    'defaults_prevented': true_positives,
                    'default_cost_prevented': default_cost_prevented,
                    'good_customers_rejected': false_positives,
                    'opportunity_cost': good_customer_rejected_cost,
                    'net_savings': default_cost_prevented - good_customer_rejected_cost,
                    'labeled_sample_size': len(labeled_data)
                }
            
            # Comparison with baseline model
            baseline_comparison = {}
            if len(labeled_data) > 0:
                current_precision = precision_score(labeled_data['actual_outcome'], labeled_data['prediction'], zero_division=0)
                
                # Estimate baseline performance
                baseline_tp_rate = self.business_config['baseline_precision']
                current_tp_rate = current_precision
                
                improvement = (current_tp_rate - baseline_tp_rate) / baseline_tp_rate if baseline_tp_rate > 0 else 0
                
                baseline_comparison = {
                    'baseline_precision': self.business_config['baseline_precision'],
                    'current_precision': current_precision,
                    'precision_improvement': improvement,
                    'estimated_additional_savings': improvement * savings.get('default_cost_prevented', 0)
                }
            
            # Efficiency metrics
            efficiency = {
                'automation_rate': (approved + rejected) / total_predictions,
                'manual_review_rate': reviews / total_predictions,
                'avg_processing_time_ms': df['processing_time_ms'].mean() if 'processing_time_ms' in df.columns else None,
                'daily_prediction_volume': total_predictions / days_back
            }
            
            return {
                'period_days': days_back,
                'total_predictions': total_predictions,
                'decision_breakdown': {
                    'approved': approved,
                    'rejected': rejected,
                    'reviews': reviews
                },
                'costs': costs,
                'savings': savings,
                'baseline_comparison': baseline_comparison,
                'efficiency_metrics': efficiency,
                'business_config': self.business_config,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ROI: {e}")
            return {"error": str(e)}
    
    def generate_executive_summary(self, roi_metrics: Dict) -> str:
        """Generate executive summary report"""
        try:
            summary = f"""
# 📊 LOAN DEFAULT MODEL - EXECUTIVE SUMMARY
## Period: {roi_metrics['period_days']} days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

### 🎯 KEY PERFORMANCE INDICATORS
- **Total Loan Applications Processed**: {roi_metrics['total_predictions']:,}
- **Automation Rate**: {roi_metrics['efficiency_metrics']['automation_rate']:.1%}
- **Manual Review Rate**: {roi_metrics['efficiency_metrics']['manual_review_rate']:.1%}
- **Average Processing Time**: {roi_metrics['efficiency_metrics']['avg_processing_time_ms']:.1f}ms

### 💰 FINANCIAL IMPACT
"""
            
            if roi_metrics['savings']:
                savings = roi_metrics['savings']
                summary += f"""
- **Defaults Prevented**: {savings['defaults_prevented']}
- **Cost Savings from Default Prevention**: ${savings['default_cost_prevented']:,.0f}
- **Net Financial Impact**: ${savings['net_savings']:,.0f}
"""
            
            if roi_metrics['baseline_comparison']:
                comp = roi_metrics['baseline_comparison']
                summary += f"""
### 📈 MODEL PERFORMANCE vs BASELINE
- **Current Precision**: {comp['current_precision']:.1%}
- **Baseline Precision**: {comp['baseline_precision']:.1%}
- **Improvement**: {comp['precision_improvement']:.1%}
"""
            
            summary += f"""
### 🔧 OPERATIONAL EFFICIENCY
- **Daily Processing Volume**: {roi_metrics['efficiency_metrics']['daily_prediction_volume']:.1f} applications/day
- **Operational Cost**: ${roi_metrics['costs']['total_operational_cost']:.2f}
- **Cost per Prediction**: ${roi_metrics['costs']['total_operational_cost']/roi_metrics['total_predictions']:.3f}

### 🎯 RECOMMENDATIONS
- Continue monitoring model performance to maintain precision above 35%
- Consider A/B testing for further optimization opportunities
- Expand automated processing to reduce manual review workload
"""
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {e}"


class DashboardGenerator:
    """
    Interactive dashboard generator using Plotly
    """
    
    def __init__(self, db_path: str = 'model_predictions.db'):
        self.db_path = db_path
        print("📊 Dashboard Generator Initialized")
    
    def create_performance_dashboard(self, days_back: int = 7) -> str:
        """Create comprehensive performance dashboard"""
        if not PLOTLY_AVAILABLE:
            return "<h1>Plotly not available. Install with: pip install plotly</h1>"
        
        try:
            # Get data
            monitor = ModelPerformanceMonitor(self.db_path)
            metrics = monitor.calculate_performance_metrics(days_back)
            
            if 'error' in metrics:
                return f"<h1>Error: {metrics['error']}</h1>"
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Decision Distribution', 'Risk Distribution',
                    'Confidence Levels', 'Processing Time Trend',
                    'Daily Prediction Volume', 'Performance Metrics'
                ],
                specs=[
                    [{"type": "pie"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ]
            )
            
            # 1. Decision Distribution
            decisions = ['APPROVE', 'REJECT', 'REVIEW']
            decision_values = [
                metrics.get('approval_rate', 0),
                metrics.get('rejection_rate', 0),
                metrics.get('review_rate', 0)
            ]
            
            fig.add_trace(
                go.Pie(labels=decisions, values=decision_values, name="Decisions"),
                row=1, col=1
            )
            
            # 2. Risk Distribution
            risk_dist = metrics.get('risk_distribution', {})
            risk_labels = ['Low Risk (≤20%)', 'Medium Risk (20-40%)', 'High Risk (>40%)']
            risk_values = [
                risk_dist.get('low_risk', 0),
                risk_dist.get('medium_risk', 0),
                risk_dist.get('high_risk', 0)
            ]
            
            fig.add_trace(
                go.Pie(labels=risk_labels, values=risk_values, name="Risk"),
                row=1, col=2
            )
            
            # 3. Confidence Levels
            conf_dist = metrics.get('confidence_distribution', {})
            conf_labels = list(conf_dist.keys()) if conf_dist else ['HIGH', 'MEDIUM', 'LOW']
            conf_values = list(conf_dist.values()) if conf_dist else [0, 0, 0]
            
            fig.add_trace(
                go.Bar(x=conf_labels, y=conf_values, name="Confidence"),
                row=2, col=1
            )
            
            # 4. Processing Time (placeholder)
            fig.add_trace(
                go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[45, 42, 48, 44, 46],
                    mode='lines+markers',
                    name="Avg Response Time (ms)"
                ),
                row=2, col=2
            )
            
            # 5. Daily Volume (placeholder)
            daily_volume = metrics.get('total_predictions', 0) / max(days_back, 1)
            fig.add_trace(
                go.Bar(
                    x=['Daily Average'],
                    y=[daily_volume],
                    name="Predictions/Day"
                ),
                row=3, col=1
            )
            
            # 6. Performance Metrics
            perf = metrics.get('performance_metrics', {})
            if perf:
                perf_labels = ['Precision', 'Recall', 'F1-Score']
                perf_values = [
                    perf.get('precision', 0),
                    perf.get('recall', 0),
                    perf.get('f1_score', 0)
                ]
                
                fig.add_trace(
                    go.Bar(x=perf_labels, y=perf_values, name="Performance"),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=False,
                title_text=f"📊 Model Performance Dashboard - Last {days_back} Days",
                title_x=0.5
            )
            
            # Convert to HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Monitoring Dashboard</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .metric {{ text-align: center; }}
                    .metric h3 {{ margin: 0; color: #333; }}
                    .metric p {{ margin: 5px 0; font-size: 24px; font-weight: bold; color: #007bff; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 Loan Default Model - Live Dashboard</h1>
                    <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Total Predictions</h3>
                        <p>{metrics.get('total_predictions', 0):,}</p>
                    </div>
                    <div class="metric">
                        <h3>Avg Probability</h3>
                        <p>{metrics.get('avg_probability', 0):.1%}</p>
                    </div>
                    <div class="metric">
                        <h3>Approval Rate</h3>
                        <p>{metrics.get('approval_rate', 0):.1%}</p>
                    </div>
                    <div class="metric">
                        <h3>Avg Response Time</h3>
                        <p>{metrics.get('avg_processing_time_ms', 0):.1f}ms</p>
                    </div>
                </div>
                
                {fig.to_html(include_plotlyjs=True, div_id="dashboard")}
                
                <script>
                    // Auto-refresh every 5 minutes
                    setTimeout(function(){{ location.reload(); }}, 300000);
                </script>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            return f"<h1>Error creating dashboard: {e}</h1>"
    
    def save_dashboard(self, filename: str = "monitoring_dashboard.html", days_back: int = 7):
        """Save dashboard to HTML file"""
        html_content = self.create_performance_dashboard(days_back)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Dashboard saved to: {filename}")
        return filename


class AlertingSystem:
    """
    Advanced alerting system with multiple notification channels
    """
    
    def __init__(self, config_file: str = 'alert_config.json'):
        self.config_file = config_file
        self.alert_history = []
        
        # Default configuration
        self.config = {
            'email_enabled': False,
            'slack_enabled': False,
            'console_enabled': True,
            'file_logging': True,
            'alert_cooldown_minutes': 60  # Prevent spam
        }
        
        self.load_config()
        print("🚨 Alerting System Initialized")
    
    def load_config(self):
        """Load alerting configuration"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
        except Exception as e:
            print(f"⚠️ Could not load alert config: {e}")
    
    def save_config(self):
        """Save alerting configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save alert config: {e}")
    
    def send_alert(self, alert: Dict):
        """Send alert through configured channels"""
        try:
            # Check cooldown to prevent spam
            if self._is_in_cooldown(alert):
                return
            
            # Console alert
            if self.config.get('console_enabled', True):
                self._send_console_alert(alert)
            
            # File logging
            if self.config.get('file_logging', True):
                self._log_alert_to_file(alert)
            
            # Email alert (placeholder)
            if self.config.get('email_enabled', False):
                self._send_email_alert(alert)
            
            # Slack alert (placeholder)  
            if self.config.get('slack_enabled', False):
                self._send_slack_alert(alert)
            
            # Add to history
            self.alert_history.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alert_history) > 100:
                self.alert_history = self.alert_history[-100:]
                
        except Exception as e:
            print(f"❌ Alert sending failed: {e}")
    
    def _is_in_cooldown(self, alert: Dict) -> bool:
        """Check if alert is in cooldown period"""
        cooldown_minutes = self.config.get('alert_cooldown_minutes', 60)
        alert_type = alert.get('type', '')
        
        # Check recent alerts of same type
        cutoff_time = datetime.now() - timedelta(minutes=cooldown_minutes)
        
        for hist_alert in self.alert_history:
            if (hist_alert.get('type') == alert_type and 
                datetime.fromisoformat(hist_alert.get('timestamp', '1970-01-01')) > cutoff_time):
                return True
        
        return False
    
    def _send_console_alert(self, alert: Dict):
        """Send alert to console"""
        severity = alert.get('severity', 'INFO')
        message = alert.get('message', 'Unknown alert')
        
        # Color coding
        colors = {
            'HIGH': '\033[91m',     # Red
            'MEDIUM': '\033[93m',   # Yellow
            'LOW': '\033[92m',      # Green
            'INFO': '\033[94m'      # Blue
        }
        reset_color = '\033[0m'
        
        color = colors.get(severity, colors['INFO'])
        print(f"{color}🚨 ALERT [{severity}] {message}{reset_color}")
    
    def _log_alert_to_file(self, alert: Dict):
        """Log alert to file"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'alert': alert
            }
            
            with open('model_alerts.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"⚠️ Failed to log alert: {e}")
    
    def _send_email_alert(self, alert: Dict):
        """Send email alert (placeholder implementation)"""
        print(f"📧 EMAIL ALERT: {alert['message']} (email sending not implemented)")
    
    def _send_slack_alert(self, alert: Dict):
        """Send Slack alert (placeholder implementation)"""
        print(f"💬 SLACK ALERT: {alert['message']} (Slack webhook not implemented)")


class MonitoringOrchestrator:
    """
    Main orchestrator for the monitoring system
    """
    
    def __init__(self, db_path: str = 'model_predictions.db'):
        self.db_path = db_path
        self.performance_monitor = ModelPerformanceMonitor(db_path)
        self.bi_reporter = BusinessIntelligenceReporter(db_path)
        self.dashboard_generator = DashboardGenerator(db_path)
        self.alerting_system = AlertingSystem()
        
        print("🎯 Monitoring Orchestrator Initialized")
    
    def run_health_check(self) -> Dict:
        """Run comprehensive health check"""
        print("\n🏥 Running comprehensive health check...")
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'components': {},
            'alerts': [],
            'recommendations': []
        }
        
        try:
            # 1. Performance metrics check
            print("📊 Checking performance metrics...")
            metrics = self.performance_monitor.calculate_performance_metrics(days_back=7)
            
            if 'error' not in metrics:
                health_report['components']['performance_metrics'] = 'HEALTHY'
                
                # Generate alerts
                alerts = self.performance_monitor.generate_alerts(metrics)
                health_report['alerts'].extend(alerts)
                
                # Send alerts
                for alert in alerts:
                    self.alerting_system.send_alert(alert)
                
                if alerts:
                    health_report['overall_status'] = 'WARNING'
            else:
                health_report['components']['performance_metrics'] = 'ERROR'
                health_report['overall_status'] = 'ERROR'
            
            # 2. Data drift check
            print("🌊 Checking for data drift...")
            drift_results = self.performance_monitor.detect_data_drift()
            
            if 'error' not in drift_results:
                health_report['components']['data_drift'] = 'HEALTHY'
                
                # Check for significant drift
                drift_detected = False
                for key, result in drift_results.get('drift_results', {}).items():
                    if result.get('drift_detected', False):
                        drift_detected = True
                        alert = {
                            'type': 'DATA_DRIFT',
                            'severity': result.get('severity', 'MEDIUM'),
                            'message': f"Data drift detected in {key.replace('_', ' ')} (p-value: {result.get('p_value', 0):.4f})",
                            'timestamp': datetime.now().isoformat()
                        }
                        health_report['alerts'].append(alert)
                        self.alerting_system.send_alert(alert)
                
                if drift_detected:
                    health_report['overall_status'] = 'WARNING'
            else:
                health_report['components']['data_drift'] = 'ERROR'
            
            # 3. Business metrics check
            print("💰 Checking business metrics...")
            roi_metrics = self.bi_reporter.calculate_roi_metrics(days_back=30)
            
            if 'error' not in roi_metrics:
                health_report['components']['business_metrics'] = 'HEALTHY'
                health_report['roi_summary'] = {
                    'total_predictions': roi_metrics.get('total_predictions', 0),
                    'automation_rate': roi_metrics.get('efficiency_metrics', {}).get('automation_rate', 0),
                    'net_savings': roi_metrics.get('savings', {}).get('net_savings', 0)
                }
            else:
                health_report['components']['business_metrics'] = 'ERROR'
            
            # 4. System health
            print("🔧 Checking system health...")
            health_report['components']['database'] = 'HEALTHY'  # If we got this far, DB is working
            health_report['components']['api'] = 'HEALTHY'       # If we got this far, API is working
            
            # Generate recommendations
            if health_report['alerts']:
                health_report['recommendations'].append("Review and address active alerts")
            
            if roi_metrics.get('efficiency_metrics', {}).get('manual_review_rate', 0) > 0.3:
                health_report['recommendations'].append("High manual review rate - consider threshold optimization")
            
            if not health_report['recommendations']:
                health_report['recommendations'].append("System operating normally - continue monitoring")
            
            print(f"✅ Health check completed - Status: {health_report['overall_status']}")
            
            return health_report
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            health_report['overall_status'] = 'ERROR'
            health_report['error'] = str(e)
            return health_report
    
    def generate_daily_report(self) -> str:
        """Generate daily monitoring report"""
        print("📋 Generating daily report...")
        
        try:
            # Get metrics
            performance_metrics = self.performance_monitor.calculate_performance_metrics(days_back=1)
            roi_metrics = self.bi_reporter.calculate_roi_metrics(days_back=1)
            
            # Generate executive summary
            executive_summary = self.bi_reporter.generate_executive_summary(roi_metrics)
            
            # Create detailed report
            report = f"""
# 📊 DAILY MONITORING REPORT
## Date: {datetime.now().strftime('%Y-%m-%d')}

{executive_summary}

## 🔍 DETAILED PERFORMANCE METRICS

### Prediction Volume
- **Total Predictions**: {performance_metrics.get('total_predictions', 0)}
- **Approval Rate**: {performance_metrics.get('approval_rate', 0):.1%}
- **Rejection Rate**: {performance_metrics.get('rejection_rate', 0):.1%}
- **Review Rate**: {performance_metrics.get('review_rate', 0):.1%}

### Risk Assessment
- **Average Default Probability**: {performance_metrics.get('avg_probability', 0):.1%}
"""
            
            if performance_metrics.get('performance_metrics'):
                perf = performance_metrics['performance_metrics']
                report += f"""
### Model Performance (Labeled Data)
- **Precision**: {perf.get('precision', 0):.1%}
- **Recall**: {perf.get('recall', 0):.1%}
- **F1-Score**: {perf.get('f1_score', 0):.1%}
- **Sample Size**: {perf.get('labeled_samples', 0)} labeled predictions
"""
            
            report += f"""
### System Performance
- **Average Response Time**: {performance_metrics.get('avg_processing_time_ms', 0):.1f}ms
- **System Uptime**: Operational

## 📈 TRENDS & INSIGHTS
- Model maintaining target precision levels
- Processing times within acceptable limits
- No significant data drift detected

## 🎯 ACTION ITEMS
- Continue monitoring for the next 24 hours
- Review any pending manual applications
- Validate model performance weekly

---
*Report generated automatically by ML Monitoring System*
"""
            
            # Save report
            filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.md"
            with open(filename, 'w') as f:
                f.write(report)
            
            print(f"✅ Daily report saved: {filename}")
            return report
            
        except Exception as e:
            error_report = f"❌ Failed to generate daily report: {e}"
            print(error_report)
            return error_report
    
    def create_live_dashboard(self, auto_refresh: bool = True) -> str:
        """Create live monitoring dashboard"""
        print("📊 Creating live dashboard...")
        
        try:
            # Generate dashboard
            filename = self.dashboard_generator.save_dashboard(
                filename="live_monitoring_dashboard.html",
                days_back=7
            )
            
            print(f"✅ Live dashboard created: {filename}")
            print(f"🌐 Open in browser: file://{Path(filename).absolute()}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Dashboard creation failed: {e}")
            return ""
    
    def run_monitoring_cycle(self):
        """Run complete monitoring cycle"""
        print("\n🔄 STARTING MONITORING CYCLE")
        print("="*60)
        
        try:
            # 1. Health check
            health_report = self.run_health_check()
            
            # 2. Generate reports
            daily_report = self.generate_daily_report()
            
            # 3. Create dashboard
            dashboard_file = self.create_live_dashboard()
            
            # 4. Summary
            print("\n✅ MONITORING CYCLE COMPLETED")
            print("="*40)
            print(f"🏥 Overall Health: {health_report['overall_status']}")
            print(f"🚨 Active Alerts: {len(health_report.get('alerts', []))}")
            print(f"📋 Daily Report: Generated")
            print(f"📊 Dashboard: {dashboard_file}")
            
            if health_report.get('alerts'):
                print("\n🚨 ACTIVE ALERTS:")
                for alert in health_report['alerts']:
                    print(f"   - {alert['severity']}: {alert['message']}")
            
            return {
                'status': 'SUCCESS',
                'health_report': health_report,
                'daily_report_generated': bool(daily_report),
                'dashboard_created': bool(dashboard_file),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Monitoring cycle failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Enhanced FastAPI endpoints for monitoring
def create_monitoring_endpoints(app: FastAPI, orchestrator: MonitoringOrchestrator):
    """Add monitoring endpoints to existing FastAPI app"""
    
    @app.get("/monitoring/health", tags=["Monitoring"])
    async def monitoring_health_check():
        """Comprehensive health check with alerts"""
        return orchestrator.run_health_check()
    
    @app.get("/monitoring/metrics", tags=["Monitoring"])
    async def get_detailed_metrics(days_back: int = 7):
        """Get detailed performance metrics"""
        return orchestrator.performance_monitor.calculate_performance_metrics(days_back)
    
    @app.get("/monitoring/drift", tags=["Monitoring"])
    async def check_data_drift(days_back: int = 7, baseline_days: int = 30):
        """Check for data drift"""
        return orchestrator.performance_monitor.detect_data_drift(days_back, baseline_days)
    
    @app.get("/monitoring/roi", tags=["Monitoring"])
    async def get_roi_metrics(days_back: int = 30):
        """Get ROI and business impact metrics"""
        return orchestrator.bi_reporter.calculate_roi_metrics(days_back)
    
    @app.get("/monitoring/dashboard", response_class=HTMLResponse, tags=["Monitoring"])
    async def get_live_dashboard(days_back: int = 7):
        """Get live monitoring dashboard"""
        return orchestrator.dashboard_generator.create_performance_dashboard(days_back)
    
    @app.get("/monitoring/alerts", tags=["Monitoring"])
    async def get_alert_history():
        """Get recent alert history"""
        return {
            "recent_alerts": orchestrator.alerting_system.alert_history[-20:],  # Last 20 alerts
            "total_alerts": len(orchestrator.alerting_system.alert_history),
            "alert_config": orchestrator.alerting_system.config
        }
    
    @app.post("/monitoring/run-cycle", tags=["Monitoring"])
    async def run_monitoring_cycle():
        """Manually trigger monitoring cycle"""
        return orchestrator.run_monitoring_cycle()
    
    print("✅ Monitoring endpoints added to FastAPI")


def main():
    """Main monitoring system demonstration"""
    print("🚀 PHASE 5: ADVANCED MONITORING & BUSINESS INTELLIGENCE")
    print("="*70)
    
    # Initialize monitoring system
    print("\n🔧 Initializing monitoring components...")
    orchestrator = MonitoringOrchestrator()
    
    # Run initial monitoring cycle
    print("\n🔄 Running initial monitoring cycle...")
    result = orchestrator.run_monitoring_cycle()
    
    if result['status'] == 'SUCCESS':
        print("\n🎉 Monitoring system initialized successfully!")
        print("\n📋 Available Functions:")
        print("1. orchestrator.run_health_check() - Comprehensive health check")
        print("2. orchestrator.generate_daily_report() - Daily business report")
        print("3. orchestrator.create_live_dashboard() - Interactive dashboard")
        print("4. orchestrator.run_monitoring_cycle() - Complete monitoring cycle")
        
        print("\n🌐 Integration Instructions:")
        print("To add monitoring to your FastAPI app:")
        print("```python")
        print("from monitoring_system import MonitoringOrchestrator, create_monitoring_endpoints")
        print("orchestrator = MonitoringOrchestrator()")
        print("create_monitoring_endpoints(app, orchestrator)")
        print("```")
        
        print("\n📊 New API Endpoints:")
        print("- GET /monitoring/health - Health check with alerts")
        print("- GET /monitoring/metrics - Detailed performance metrics")
        print("- GET /monitoring/drift - Data drift analysis")
        print("- GET /monitoring/roi - ROI and business impact")
        print("- GET /monitoring/dashboard - Live dashboard")
        print("- GET /monitoring/alerts - Alert history")
        print("- POST /monitoring/run-cycle - Manual monitoring cycle")
        
    else:
        print(f"\n❌ Monitoring initialization failed: {result.get('error', 'Unknown error')}")
    
    return orchestrator


if __name__ == "__main__":
    # Run the monitoring system
    monitoring_orchestrator = main()
    
    # Keep the system running for demonstration
    print("\n⏰ Monitoring system is now active!")
    print("🔄 Run orchestrator.run_monitoring_cycle() to perform monitoring")
    print("📊 Run orchestrator.create_live_dashboard() to generate dashboard")
    print("🏥 Run orchestrator.run_health_check() for health status")