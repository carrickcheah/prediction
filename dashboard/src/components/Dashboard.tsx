import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Alert as MuiAlert,
  CircularProgress,
  Tab,
  Tabs,
  Card,
  CardContent
} from '@mui/material';
import { api, AnalyticsSummary } from '../api/client';
import PredictionChart from './PredictionChart';
import AlertsPanel from './AlertsPanel';
import PartSearch from './PartSearch';
import StatsCards from './StatsCards';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Dashboard: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [selectedPartId, setSelectedPartId] = useState<number | null>(null);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      const data = await api.getAnalyticsSummary();
      setAnalytics(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load analytics');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handlePartSelect = (partId: number) => {
    setSelectedPartId(partId);
    setTabValue(1); // Switch to predictions tab
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <MuiAlert severity="error">{error}</MuiAlert>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" gutterBottom component="h1" sx={{ mb: 4 }}>
        Inventory Forecasting Dashboard
      </Typography>

      {analytics && <StatsCards analytics={analytics} />}

      <Paper sx={{ mt: 4 }}>
        <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tab label="Overview" />
          <Tab label="Predictions" />
          <Tab label="Alerts" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    System Status
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="body1" color="text.secondary">
                        Total Models Loaded: {analytics?.total_models || 0}
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Average Model Accuracy: {((analytics?.average_accuracy || 0) * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Last Training: {analytics?.last_training_date || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6" gutterBottom>
                        Most Urgent Parts
                      </Typography>
                      {analytics?.most_urgent_parts?.map((part) => (
                        <Box
                          key={part.part_id}
                          sx={{
                            p: 1,
                            mb: 1,
                            bgcolor: 'background.paper',
                            border: 1,
                            borderColor: 'divider',
                            borderRadius: 1,
                            cursor: 'pointer',
                            '&:hover': { bgcolor: 'action.hover' }
                          }}
                          onClick={() => handlePartSelect(part.part_id)}
                        >
                          <Typography variant="body2">
                            {part.stock_code} - Risk: {(part.stockout_risk * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                      ))}
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <PartSearch onPartSelect={setSelectedPartId} />
            </Grid>
            {selectedPartId && (
              <Grid item xs={12}>
                <PredictionChart partId={selectedPartId} />
              </Grid>
            )}
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <AlertsPanel />
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Dashboard;