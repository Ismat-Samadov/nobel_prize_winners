import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Typography,
  Paper,
  Box,
  Button,
  CircularProgress,
  Alert,
  Grid,
  Card,
  CardContent,
  Chip,
} from '@mui/material';
import {
  Assessment as ResultsIcon,
  TrendingUp as TrendingIcon,
  Download as DownloadIcon,
  ArrowBack as BackIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { nerAPI } from '../services/api';
import toast from 'react-hot-toast';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const ResultsPage = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadResults();
  }, [sessionId]);

  const loadResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await nerAPI.getResults(sessionId);
      setResults(data);
    } catch (err) {
      console.error('Failed to load results:', err);
      setError('Failed to load training results');
      toast.error('Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatMetric = (value) => {
    return value ? value.toFixed(3) : '0.000';
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 3 }}>
        {error}
      </Alert>
    );
  }

  if (!results) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          No Results Available
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Results will be available after training completion.
        </Typography>
        <Button variant="contained" onClick={() => navigate('/sessions')}>
          Back to Sessions
        </Button>
      </Paper>
    );
  }

  // Prepare chart data
  const learningCurveData = results.learning_curve?.rounds?.map((round, index) => ({
    round: round,
    f1: results.learning_curve.f1_scores[index],
    precision: results.learning_curve.precision_scores[index],
    recall: results.learning_curve.recall_scores[index],
  })) || [];

  const performanceData = results.final_performance ? [
    { name: 'F1 Score', value: results.final_performance.f1 },
    { name: 'Precision', value: results.final_performance.precision },
    { name: 'Recall', value: results.final_performance.recall },
  ] : [];

  const efficiencyData = results.sample_efficiency ? [
    {
      name: 'Sample Efficiency',
      labeled: results.sample_efficiency.labeled_samples,
      total: results.sample_efficiency.total_samples,
      efficiency: results.sample_efficiency.efficiency_ratio,
    },
  ] : [];

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center' }}>
            <ResultsIcon sx={{ mr: 2, fontSize: 'inherit' }} />
            Training Results
          </Typography>
          <Box>
            <Button
              variant="outlined"
              startIcon={<BackIcon />}
              onClick={() => navigate('/sessions')}
              sx={{ mr: 2 }}
            >
              Back to Sessions
            </Button>
            <Button
              variant="contained"
              startIcon={<DownloadIcon />}
              onClick={() => toast.info('Download feature coming soon')}
            >
              Download Report
            </Button>
          </Box>
        </Box>
        <Typography variant="body1" color="text.secondary">
          Session: {sessionId.substring(0, 8)}... | Status: {results.status}
        </Typography>
      </Paper>

      {/* Performance Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {formatMetric(results.final_performance?.f1)}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                F1 Score
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="secondary">
                {formatMetric(results.final_performance?.precision)}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Precision
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                {formatMetric(results.final_performance?.recall)}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Recall
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {formatPercentage(results.sample_efficiency?.efficiency_ratio || 0)}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Sample Efficiency
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        {/* Learning Curve */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <TrendingIcon sx={{ mr: 1 }} />
              Learning Curve
            </Typography>
            <Box sx={{ height: 400 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={learningCurveData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="round" 
                    label={{ value: 'Training Round', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    label={{ value: 'Score', angle: -90, position: 'insideLeft' }}
                    domain={[0, 1]}
                  />
                  <Tooltip formatter={(value) => formatMetric(value)} />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="f1" 
                    stroke="#8884d8" 
                    name="F1 Score"
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="precision" 
                    stroke="#82ca9d" 
                    name="Precision"
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="recall" 
                    stroke="#ffc658" 
                    name="Recall"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Performance Distribution */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Performance Metrics
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={performanceData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${formatMetric(value)}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {performanceData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => formatMetric(value)} />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Sample Efficiency */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Sample Efficiency
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={efficiencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis label={{ value: 'Samples', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="labeled" fill="#8884d8" name="Labeled Samples" />
                  <Bar dataKey="total" fill="#82ca9d" name="Total Samples" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Training Summary */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Training Summary
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body1">Total Samples:</Typography>
                <Chip 
                  label={results.sample_efficiency?.total_samples || 'N/A'} 
                  color="primary" 
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body1">Labeled Samples:</Typography>
                <Chip 
                  label={results.sample_efficiency?.labeled_samples || 'N/A'} 
                  color="secondary" 
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body1">Efficiency Ratio:</Typography>
                <Chip 
                  label={formatPercentage(results.sample_efficiency?.efficiency_ratio || 0)} 
                  color="success" 
                />
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body1">Training Rounds:</Typography>
                <Chip 
                  label={results.learning_curve?.rounds?.length || 'N/A'} 
                  color="warning" 
                />
              </Box>
            </Box>

            <Box sx={{ mt: 3 }}>
              <Typography variant="body2" color="text.secondary">
                This active learning session achieved {formatMetric(results.final_performance?.f1 || 0)} F1 score 
                using only {formatPercentage(results.sample_efficiency?.efficiency_ratio || 0)} of the available data.
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Action Buttons */}
      <Paper sx={{ p: 3, mt: 3, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          Next Steps
        </Typography>
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            onClick={() => navigate('/predict')}
          >
            Test Model Predictions
          </Button>
          <Button
            variant="outlined"
            onClick={() => navigate('/train')}
          >
            Start New Training
          </Button>
          <Button
            variant="outlined"
            onClick={() => navigate('/sessions')}
          >
            View All Sessions
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default ResultsPage;