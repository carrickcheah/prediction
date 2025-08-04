import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  CircularProgress,
  Alert as MuiAlert,
  List,
  ListItem,
  ListItemText,
  Chip,
  IconButton,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import { api, Alert } from '../api/client';
import { format } from 'date-fns';

const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [urgencyFilter, setUrgencyFilter] = useState<string>('all');

  useEffect(() => {
    loadAlerts();
  }, [urgencyFilter]);

  const loadAlerts = async () => {
    try {
      setLoading(true);
      setError(null);
      const filter = urgencyFilter === 'all' ? undefined : urgencyFilter;
      const data = await api.getAlerts(filter);
      setAlerts(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load alerts');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    try {
      const criticalParts = alerts
        .filter(a => a.urgency === 'critical')
        .map(a => a.part_id);
      
      const report = await api.generateReport(criticalParts, 'excel');
      
      // Download the report
      if (report.download_url) {
        const fileName = report.download_url.split('/').pop();
        if (fileName) {
          await api.downloadReport(fileName);
        }
      }
    } catch (err: any) {
      setError('Failed to generate report');
    }
  };

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      default: return 'success';
    }
  };

  const getAlertTypeIcon = (type: string) => {
    switch (type) {
      case 'stockout_imminent': return '⚠️';
      case 'high_consumption': return '📈';
      case 'supply_delay': return '🚚';
      default: return '📊';
    }
  };

  if (loading && alerts.length === 0) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5">
          Inventory Alerts
        </Typography>
        <Box display="flex" gap={2} alignItems="center">
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Urgency</InputLabel>
            <Select
              value={urgencyFilter}
              label="Urgency"
              onChange={(e) => setUrgencyFilter(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="critical">Critical</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="low">Low</MenuItem>
            </Select>
          </FormControl>
          <IconButton onClick={loadAlerts} disabled={loading}>
            <RefreshIcon />
          </IconButton>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleGenerateReport}
            disabled={alerts.length === 0}
          >
            Export Report
          </Button>
        </Box>
      </Box>

      {error && (
        <MuiAlert severity="error" sx={{ mb: 2 }}>
          {error}
        </MuiAlert>
      )}

      {alerts.length === 0 ? (
        <Typography color="text.secondary">
          No alerts found for the selected filter
        </Typography>
      ) : (
        <Grid container spacing={2}>
          {alerts.map((alert, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                    <Box>
                      <Typography variant="h6" component="div">
                        {alert.stock_code}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Part ID: {alert.part_id}
                      </Typography>
                    </Box>
                    <Chip
                      label={alert.urgency}
                      color={getUrgencyColor(alert.urgency)}
                      size="small"
                    />
                  </Box>
                  
                  <Box display="flex" alignItems="center" gap={1} mb={1}>
                    <Typography variant="body2">
                      {getAlertTypeIcon(alert.alert_type)}
                    </Typography>
                    <Typography variant="body2" fontWeight="medium">
                      {alert.alert_type.replace(/_/g, ' ').toUpperCase()}
                    </Typography>
                  </Box>

                  <Typography variant="body2" sx={{ mb: 2 }}>
                    {alert.message}
                  </Typography>

                  {alert.predicted_stockout_date && (
                    <Typography variant="body2" color="error" sx={{ mb: 1 }}>
                      Stockout: {format(new Date(alert.predicted_stockout_date), 'MMM dd, yyyy')}
                    </Typography>
                  )}

                  {alert.current_consumption_rate !== undefined && (
                    <Typography variant="body2" color="text.secondary">
                      Daily consumption: {alert.current_consumption_rate.toFixed(2)} units
                    </Typography>
                  )}
                </CardContent>
                <CardActions>
                  <Typography variant="body2" color="primary" sx={{ px: 1 }}>
                    Action: {alert.recommended_action}
                  </Typography>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Paper>
  );
};

export default AlertsPanel;