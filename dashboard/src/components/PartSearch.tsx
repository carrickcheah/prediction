import React, { useState } from 'react';
import {
  TextField,
  Button,
  Box,
  Paper,
  Typography,
  Grid
} from '@mui/material';

interface PartSearchProps {
  onPartSelect: (partId: number) => void;
}

const PartSearch: React.FC<PartSearchProps> = ({ onPartSelect }) => {
  const [partId, setPartId] = useState('');
  
  // Common parts for quick access
  const commonParts = [
    { id: 1000045, code: 'CP08-415B' },
    { id: 1000074, code: 'D1.8-SWMB' },
    { id: 1000087, code: 'D4.0-SWMB' },
    { id: 1000205, code: 'T0.4-SGCC' },
    { id: 1000290, code: 'T0.7-SGCC' },
    { id: 1000332, code: 'T0.8-SGCC' }
  ];

  const handleSearch = () => {
    const id = parseInt(partId);
    if (!isNaN(id)) {
      onPartSelect(id);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Search Parts
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Enter Part ID"
              value={partId}
              onChange={(e) => setPartId(e.target.value)}
              onKeyPress={handleKeyPress}
              type="number"
              variant="outlined"
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              fullWidth
              variant="contained"
              onClick={handleSearch}
              disabled={!partId}
              size="large"
            >
              Search
            </Button>
          </Grid>
        </Grid>
      </Box>

      <Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Quick Access - Common Parts:
        </Typography>
        <Box display="flex" gap={1} flexWrap="wrap">
          {commonParts.map((part) => (
            <Button
              key={part.id}
              variant="outlined"
              size="small"
              onClick={() => onPartSelect(part.id)}
            >
              {part.code} ({part.id})
            </Button>
          ))}
        </Box>
      </Box>
    </Paper>
  );
};

export default PartSearch;