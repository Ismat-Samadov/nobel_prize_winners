import React from 'react';
import {
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Psychology as PredictIcon,
  School as TrainIcon,
  Dashboard as SessionsIcon,
  Upload as UploadIcon,
  AutoAwesome as AIIcon,
  TrendingUp as TrendingIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: 'Text Prediction',
      description: 'Get instant NER predictions on your text using pre-trained models.',
      icon: <PredictIcon fontSize="large" color="primary" />,
      action: () => navigate('/predict'),
      buttonText: 'Try Prediction',
    },
    {
      title: 'Active Learning Training',
      description: 'Train custom NER models with minimal labeled data using active learning.',
      icon: <TrainIcon fontSize="large" color="secondary" />,
      action: () => navigate('/train'),
      buttonText: 'Start Training',
    },
    {
      title: 'Training Sessions',
      description: 'Monitor and manage your active learning training sessions.',
      icon: <SessionsIcon fontSize="large" color="success" />,
      action: () => navigate('/sessions'),
      buttonText: 'View Sessions',
    },
  ];

  const benefits = [
    'Reduce labeling effort by 50-80%',
    'Faster convergence to target performance',
    'Better generalization through diverse sampling',
    'Cost-effective annotation process',
    'Multiple active learning strategies',
    'Real-time performance monitoring',
  ];

  return (
    <Box>
      {/* Hero Section */}
      <Paper
        sx={{
          p: 4,
          mb: 4,
          background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
          color: 'white',
          textAlign: 'center',
        }}
      >
        <Typography variant="h3" component="h1" gutterBottom>
          Active Learning for NER
        </Typography>
        <Typography variant="h6" component="p" sx={{ mb: 3, opacity: 0.9 }}>
          Reduce labeled data requirements for Named Entity Recognition using intelligent active learning
        </Typography>
        <Button
          variant="contained"
          size="large"
          sx={{
            backgroundColor: 'white',
            color: 'primary.main',
            '&:hover': {
              backgroundColor: 'grey.100',
            },
          }}
          onClick={() => navigate('/predict')}
        >
          Get Started
        </Button>
      </Paper>

      {/* Features Grid */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardContent sx={{ flexGrow: 1, textAlign: 'center' }}>
                <Box sx={{ mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography gutterBottom variant="h5" component="h2">
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: 'center', pb: 2 }}>
                <Button
                  size="medium"
                  variant="contained"
                  onClick={feature.action}
                >
                  {feature.buttonText}
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Benefits and How it Works */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, h: '100%' }}>
            <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <TrendingIcon sx={{ mr: 1, color: 'success.main' }} />
              Benefits
            </Typography>
            <List>
              {benefits.map((benefit, index) => (
                <ListItem key={index} sx={{ py: 0.5 }}>
                  <ListItemIcon>
                    <AIIcon color="primary" />
                  </ListItemIcon>
                  <ListItemText primary={benefit} />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, h: '100%' }}>
            <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <UploadIcon sx={{ mr: 1, color: 'info.main' }} />
              How It Works
            </Typography>
            <List>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="h6" color="primary">1</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary="Upload Your Data"
                  secondary="Start with a small labeled dataset or upload unlabeled text"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="h6" color="primary">2</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary="Train Initial Model"
                  secondary="Begin with a basic model trained on available labeled data"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="h6" color="primary">3</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary="Active Selection"
                  secondary="AI selects the most informative samples for labeling"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <Typography variant="h6" color="primary">4</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary="Iterative Improvement"
                  secondary="Label selected samples and retrain for better performance"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Quick Stats */}
      <Paper sx={{ p: 3, mt: 4, textAlign: 'center' }}>
        <Typography variant="h5" gutterBottom>
          Active Learning Performance
        </Typography>
        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} sm={4}>
            <Typography variant="h3" color="primary">
              50-80%
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Reduction in labeling effort
            </Typography>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="h3" color="secondary">
              5x
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Faster convergence
            </Typography>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="h3" color="success.main">
              95%+
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Target accuracy achievable
            </Typography>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default HomePage;