import React from 'react';
import { Grid, Card, CardContent, Typography, Box } from '@mui/material';
import { AnalyticsSummary } from '../api/client';

interface StatsCardsProps {
  analytics: AnalyticsSummary;
}

const StatsCards: React.FC<StatsCardsProps> = ({ analytics }) => {
  const cards = [
    {
      title: 'Total Models',
      value: analytics.total_models,
      color: '#2196f3'
    },
    {
      title: 'Critical Alerts',
      value: analytics.critical_alerts,
      color: '#f44336'
    },
    {
      title: 'High Alerts',
      value: analytics.high_alerts,
      color: '#ff9800'
    },
    {
      title: 'Avg Accuracy',
      value: `${(analytics.average_accuracy * 100).toFixed(1)}%`,
      color: '#4caf50'
    }
  ];

  return (
    <Grid container spacing={3}>
      {cards.map((card, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="body2">
                {card.title}
              </Typography>
              <Typography variant="h4" component="div" sx={{ color: card.color }}>
                {card.value}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};

export default StatsCards;