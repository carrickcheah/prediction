import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const API_KEY = process.env.REACT_APP_API_KEY || 'development-key-123';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
  }
});

// Types
export interface PredictionResponse {
  part_id: number;
  stock_code: string;
  predictions: number[];
  dates: string[];
  confidence_interval: {
    lower: number[];
    upper: number[];
  };
  urgency: 'critical' | 'high' | 'medium' | 'low';
  stockout_risk: number;
  recommended_order_qty: number;
  model_mae: number;
  zero_percentage: number;
}

export interface AnalyticsSummary {
  total_models: number;
  critical_alerts: number;
  high_alerts: number;
  average_accuracy: number;
  last_training_date: string;
  most_urgent_parts: Array<{
    part_id: number;
    stock_code: string;
    urgency: string;
    stockout_risk: number;
  }>;
}

export interface Alert {
  part_id: number;
  stock_code: string;
  alert_type: string;
  urgency: string;
  message: string;
  recommended_action: string;
  predicted_stockout_date?: string;
  current_consumption_rate?: number;
}

// API Methods
export const api = {
  // Predictions
  getPrediction: async (partId: number, horizonDays: number = 14): Promise<PredictionResponse> => {
    const response = await apiClient.post(`/api/predict/${partId}?horizon_days=${horizonDays}`);
    return response.data;
  },

  getBatchPredictions: async (partIds: number[], horizonDays: number = 14): Promise<PredictionResponse[]> => {
    const response = await apiClient.post('/api/predict/batch', {
      part_ids: partIds,
      horizon_days: horizonDays
    });
    return response.data.predictions;
  },

  // Analytics
  getAnalyticsSummary: async (): Promise<AnalyticsSummary> => {
    const response = await apiClient.get('/api/analytics/summary');
    return response.data;
  },

  // Alerts
  getAlerts: async (urgencyFilter?: string): Promise<Alert[]> => {
    const params = urgencyFilter ? { urgency: urgencyFilter } : {};
    const response = await apiClient.get('/api/alerts', { params });
    return response.data.alerts;
  },

  // Reports
  generateReport: async (partIds?: number[], reportType: string = 'excel') => {
    const response = await apiClient.post('/api/reports/generate', {
      report_type: reportType,
      include_parts: partIds
    });
    return response.data;
  },

  downloadReport: async (fileName: string) => {
    const response = await apiClient.get(`/api/reports/download/${fileName}`, {
      responseType: 'blob'
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
    link.remove();
  }
};