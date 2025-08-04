import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import {
  Paper,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
  Grid,
  Card,
  CardContent
} from '@mui/material';
import { format, parseISO } from 'date-fns';
import { api, PredictionResponse } from '../api/client';

interface PredictionChartProps {
  partId: number;
}

const PredictionChart: React.FC<PredictionChartProps> = ({ partId }) => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (partId) {
      loadPrediction();
    }
  }, [partId]);

  const loadPrediction = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getPrediction(partId, 14);
      setPrediction(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load prediction');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  if (!prediction) {
    return <Typography>Select a part to view predictions</Typography>;
  }

  // Prepare chart data
  const chartData = prediction.dates.map((date, index) => ({
    date: format(parseISO(date), 'MMM dd'),
    fullDate: date,
    prediction: prediction.predictions[index],
    lower: prediction.confidence_interval.lower[index],
    upper: prediction.confidence_interval.upper[index]
  }));

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      default: return 'success';
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Box>
              <Typography variant="h5" gutterBottom>
                Forecast for {prediction.stock_code}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Part ID: {prediction.part_id}
              </Typography>
            </Box>
            <Box display="flex" gap={1}>
              <Chip 
                label={`Urgency: ${prediction.urgency}`}
                color={getUrgencyColor(prediction.urgency)}
                variant="filled"
              />
              <Chip 
                label={`Risk: ${(prediction.stockout_risk * 100).toFixed(0)}%`}
                variant="outlined"
              />
            </Box>
          </Box>
        </Grid>

        <Grid item xs={12} md={8}>
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload[0]) {
                    const data = payload[0].payload;
                    return (
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="body2">{data.fullDate}</Typography>
                        <Typography variant="body2" color="primary">
                          Forecast: {data.prediction.toFixed(2)}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Range: {data.lower.toFixed(2)} - {data.upper.toFixed(2)}
                        </Typography>
                      </Paper>
                    );
                  }
                  return null;
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="upper"
                stackId="1"
                stroke="none"
                fill="#e3f2fd"
                name="Upper Bound"
              />
              <Area
                type="monotone"
                dataKey="lower"
                stackId="2"
                stroke="none"
                fill="#e3f2fd"
                name="Lower Bound"
              />
              <Line
                type="monotone"
                dataKey="prediction"
                stroke="#2196f3"
                strokeWidth={3}
                dot={{ fill: '#2196f3', r: 4 }}
                activeDot={{ r: 6 }}
                name="Forecast"
              />
            </AreaChart>
          </ResponsiveContainer>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Metrics
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Model Accuracy (MAE)
                </Typography>
                <Typography variant="h6">
                  {prediction.model_mae.toFixed(3)}
                </Typography>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Zero Demand Days
                </Typography>
                <Typography variant="h6">
                  {prediction.zero_percentage.toFixed(1)}%
                </Typography>
              </Box>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Recommended Order Qty
                </Typography>
                <Typography variant="h6" color="primary">
                  {prediction.recommended_order_qty.toFixed(0)}
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  14-Day Total Forecast
                </Typography>
                <Typography variant="h6">
                  {prediction.predictions.reduce((a, b) => a + b, 0).toFixed(1)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default PredictionChart;