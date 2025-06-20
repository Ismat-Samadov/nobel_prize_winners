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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  School as TrainIcon,
  ExpandMore as ExpandMoreIcon,
  Upload as UploadIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { nerAPI } from '../services/api';
import toast from 'react-hot-toast';

const TrainPage = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [config, setConfig] = useState({
    model_name: 'bert-base-uncased',
    num_labels: 9,
    use_crf: false,
    strategy: 'uncertainty',
    uncertainty_method: 'entropy',
    num_rounds: 10,
    samples_per_round: 100,
    epochs_per_round: 3,
    batch_size: 16,
    learning_rate: 2e-5,
    initial_labeled_ratio: 0.1,
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/plain': ['.conll', '.txt'],
    },
    multiple: false,
    onDrop: async (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setLoading(true);
        try {
          const result = await nerAPI.uploadFile(file);
          setUploadedFile({ file, result });
          toast.success('File uploaded successfully!');
        } catch (error) {
          toast.error('Failed to upload file');
        } finally {
          setLoading(false);
        }
      }
    },
  });

  const handleConfigChange = (field) => (event) => {
    const value = event.target.type === 'checkbox' 
      ? event.target.checked 
      : event.target.value;
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleSliderChange = (field) => (event, value) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleStartTraining = async () => {
    setLoading(true);
    try {
      const result = await nerAPI.startTraining(config);
      toast.success('Training session started!');
      navigate(`/sessions`);
    } catch (error) {
      toast.error('Failed to start training session');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <TrainIcon sx={{ mr: 2, fontSize: 'inherit' }} />
          Start Active Learning Training
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure and start a new active learning training session to build a custom NER model with minimal labeled data.
        </Typography>
      </Paper>

      <Grid container spacing={3}>
        {/* Data Upload Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Training Data
            </Typography>
            
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.300',
                borderRadius: 2,
                p: 3,
                textAlign: 'center',
                cursor: 'pointer',
                backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                mb: 2,
              }}
            >
              <input {...getInputProps()} />
              <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                {isDragActive ? 'Drop the file here' : 'Upload Training Data'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Drag and drop a CoNLL format file, or click to select
              </Typography>
            </Box>

            {uploadedFile && (
              <Alert severity="success" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  <strong>{uploadedFile.file.name}</strong> uploaded successfully
                </Typography>
                {uploadedFile.result.stats && (
                  <Typography variant="caption" display="block">
                    {uploadedFile.result.stats.num_sentences} sentences, {uploadedFile.result.stats.num_tokens} tokens
                  </Typography>
                )}
              </Alert>
            )}

            <Button
              variant="outlined"
              onClick={() => {
                // Use sample data for demo
                setUploadedFile({
                  file: { name: 'sample_data.conll' },
                  result: { stats: { num_sentences: 1000, num_tokens: 15000 } }
                });
                toast.success('Using sample dataset for demo');
              }}
            >
              Use Sample Dataset
            </Button>
          </Paper>
        </Grid>

        {/* Configuration Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Model Configuration
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl fullWidth>
                <InputLabel>Pre-trained Model</InputLabel>
                <Select
                  value={config.model_name}
                  onChange={handleConfigChange('model_name')}
                  label="Pre-trained Model"
                >
                  <MenuItem value="bert-base-uncased">BERT Base Uncased</MenuItem>
                  <MenuItem value="distilbert-base-uncased">DistilBERT Base</MenuItem>
                  <MenuItem value="roberta-base">RoBERTa Base</MenuItem>
                </Select>
              </FormControl>

              <TextField
                label="Number of Labels"
                type="number"
                value={config.num_labels}
                onChange={handleConfigChange('num_labels')}
                fullWidth
              />

              <FormControlLabel
                control={
                  <Switch
                    checked={config.use_crf}
                    onChange={handleConfigChange('use_crf')}
                  />
                }
                label="Use CRF Layer"
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Active Learning Strategy
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControl fullWidth>
                <InputLabel>Strategy</InputLabel>
                <Select
                  value={config.strategy}
                  onChange={handleConfigChange('strategy')}
                  label="Strategy"
                >
                  <MenuItem value="random">Random Sampling</MenuItem>
                  <MenuItem value="uncertainty">Uncertainty Sampling</MenuItem>
                  <MenuItem value="committee">Query by Committee</MenuItem>
                  <MenuItem value="diversity">Diversity Sampling</MenuItem>
                  <MenuItem value="hybrid">Hybrid Sampling</MenuItem>
                </Select>
              </FormControl>

              {config.strategy === 'uncertainty' && (
                <FormControl fullWidth>
                  <InputLabel>Uncertainty Method</InputLabel>
                  <Select
                    value={config.uncertainty_method}
                    onChange={handleConfigChange('uncertainty_method')}
                    label="Uncertainty Method"
                  >
                    <MenuItem value="entropy">Entropy</MenuItem>
                    <MenuItem value="max_prob">Max Probability</MenuItem>
                    <MenuItem value="margin">Margin</MenuItem>
                    <MenuItem value="mc_dropout">Monte Carlo Dropout</MenuItem>
                  </Select>
                </FormControl>
              )}

              <Box>
                <Typography gutterBottom>
                  Initial Labeled Ratio: {(config.initial_labeled_ratio * 100).toFixed(0)}%
                </Typography>
                <Slider
                  value={config.initial_labeled_ratio}
                  onChange={handleSliderChange('initial_labeled_ratio')}
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  marks
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                />
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Advanced Configuration */}
        <Grid item xs={12}>
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Advanced Configuration</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    label="Training Rounds"
                    type="number"
                    value={config.num_rounds}
                    onChange={handleConfigChange('num_rounds')}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    label="Samples per Round"
                    type="number"
                    value={config.samples_per_round}
                    onChange={handleConfigChange('samples_per_round')}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    label="Epochs per Round"
                    type="number"
                    value={config.epochs_per_round}
                    onChange={handleConfigChange('epochs_per_round')}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <TextField
                    label="Batch Size"
                    type="number"
                    value={config.batch_size}
                    onChange={handleConfigChange('batch_size')}
                    fullWidth
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Learning Rate"
                    type="number"
                    value={config.learning_rate}
                    onChange={handleConfigChange('learning_rate')}
                    inputProps={{ step: 0.00001 }}
                    fullWidth
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, textAlign: 'center' }}>
            <Button
              variant="contained"
              size="large"
              onClick={handleStartTraining}
              disabled={loading || !uploadedFile}
              startIcon={loading ? <CircularProgress size={20} /> : <TrainIcon />}
              sx={{ mr: 2 }}
            >
              {loading ? 'Starting Training...' : 'Start Training Session'}
            </Button>
            
            <Button
              variant="outlined"
              size="large"
              onClick={() => navigate('/sessions')}
            >
              View Existing Sessions
            </Button>

            {!uploadedFile && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Please upload training data or use the sample dataset to start training.
              </Alert>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TrainPage;