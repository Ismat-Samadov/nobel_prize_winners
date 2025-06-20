import React, { useState } from 'react';
import {
  Typography,
  Paper,
  Box,
  TextField,
  Button,
  CircularProgress,
  Alert,
  Grid,
} from '@mui/material';
import { Psychology as PredictIcon } from '@mui/icons-material';
import { nerAPI } from '../services/api';
import EntityHighlighter from '../components/EntityHighlighter';
import toast from 'react-hot-toast';

const PredictPage = () => {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const sampleTexts = [
    "John Smith works at Google in Mountain View, California.",
    "Apple Inc. was founded by Steve Jobs in Cupertino.",
    "The meeting will be held at Stanford University next week.",
    "Microsoft announced a new partnership with OpenAI.",
    "Dr. Sarah Johnson from MIT will present her research on Python."
  ];

  const handlePredict = async () => {
    if (!text.trim()) {
      toast.error('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await nerAPI.predict(text);
      setPrediction(result);
      toast.success('Prediction completed successfully!');
    } catch (err) {
      console.error('Prediction error:', err);
      const errorMessage = err.response?.data?.detail || 'Failed to get prediction';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleText = (sampleText) => {
    setText(sampleText);
  };

  const handleClear = () => {
    setText('');
    setPrediction(null);
    setError(null);
  };

  return (
    <Box>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <PredictIcon sx={{ mr: 2, fontSize: 'inherit' }} />
          Named Entity Recognition Prediction
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Enter text below to identify named entities such as persons, organizations, locations, and more.
        </Typography>
      </Paper>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          {/* Input Section */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Input Text
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={6}
              variant="outlined"
              placeholder="Enter your text here for named entity recognition..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              sx={{ mb: 2 }}
            />
            
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                onClick={handlePredict}
                disabled={loading || !text.trim()}
                startIcon={loading ? <CircularProgress size={20} /> : <PredictIcon />}
              >
                {loading ? 'Analyzing...' : 'Analyze Text'}
              </Button>
              
              <Button
                variant="outlined"
                onClick={handleClear}
                disabled={loading}
              >
                Clear
              </Button>
            </Box>
          </Paper>

          {/* Results Section */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {prediction && (
            <Paper sx={{ p: 3, mb: 3 }}>
              <EntityHighlighter
                tokens={prediction.tokens}
                labels={prediction.labels}
                entities={prediction.entities}
              />
            </Paper>
          )}
        </Grid>

        {/* Sample Texts Sidebar */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Sample Texts
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Try these examples to see how the NER model works:
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {sampleTexts.map((sampleText, index) => (
                <Button
                  key={index}
                  variant="outlined"
                  size="small"
                  onClick={() => handleSampleText(sampleText)}
                  disabled={loading}
                  sx={{
                    textAlign: 'left',
                    justifyContent: 'flex-start',
                    textTransform: 'none',
                    py: 1,
                    px: 2,
                    whiteSpace: 'normal',
                    lineHeight: 1.2,
                  }}
                >
                  {sampleText}
                </Button>
              ))}
            </Box>

            {/* Entity Legend */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Entity Types
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box 
                    sx={{ 
                      width: 16, 
                      height: 16, 
                      backgroundColor: '#e3f2fd', 
                      border: '1px solid #1976d2',
                      borderRadius: 0.5 
                    }} 
                  />
                  <Typography variant="body2">PER - Person names</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box 
                    sx={{ 
                      width: 16, 
                      height: 16, 
                      backgroundColor: '#f3e5f5', 
                      border: '1px solid #7b1fa2',
                      borderRadius: 0.5 
                    }} 
                  />
                  <Typography variant="body2">ORG - Organizations</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box 
                    sx={{ 
                      width: 16, 
                      height: 16, 
                      backgroundColor: '#e8f5e8', 
                      border: '1px solid #388e3c',
                      borderRadius: 0.5 
                    }} 
                  />
                  <Typography variant="body2">LOC - Locations</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box 
                    sx={{ 
                      width: 16, 
                      height: 16, 
                      backgroundColor: '#fff3e0', 
                      border: '1px solid #f57c00',
                      borderRadius: 0.5 
                    }} 
                  />
                  <Typography variant="body2">MISC - Miscellaneous</Typography>
                </Box>
              </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PredictPage;