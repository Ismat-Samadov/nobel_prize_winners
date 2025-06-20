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
  CardActions,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Edit as AnnotateIcon,
  Save as SaveIcon,
  Skip as SkipIcon,
  CheckCircle as CompleteIcon,
} from '@mui/icons-material';
import { nerAPI } from '../services/api';
import toast from 'react-hot-toast';

const entityTypes = [
  { label: 'PER', color: '#e3f2fd', borderColor: '#1976d2', description: 'Person' },
  { label: 'ORG', color: '#f3e5f5', borderColor: '#7b1fa2', description: 'Organization' },
  { label: 'LOC', color: '#e8f5e8', borderColor: '#388e3c', description: 'Location' },
  { label: 'MISC', color: '#fff3e0', borderColor: '#f57c00', description: 'Miscellaneous' },
  { label: 'O', color: '#f5f5f5', borderColor: '#999', description: 'Outside' },
];

const AnnotatePage = () => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  
  const [session, setSession] = useState(null);
  const [samples, setSamples] = useState([]);
  const [currentSampleIndex, setCurrentSampleIndex] = useState(0);
  const [annotations, setAnnotations] = useState([]);
  const [selectedTokens, setSelectedTokens] = useState(new Set());
  const [selectedEntityType, setSelectedEntityType] = useState('PER');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [completionDialog, setCompletionDialog] = useState(false);

  useEffect(() => {
    loadSession();
  }, [sessionId]);

  const loadSession = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await nerAPI.getSession(sessionId);
      setSession(result.session);
      setSamples(result.samples_for_annotation || []);
      
      if (result.samples_for_annotation && result.samples_for_annotation.length > 0) {
        // Initialize annotations for all samples
        const initialAnnotations = result.samples_for_annotation.map(sample => ({
          id: sample.id,
          tokens: sample.tokens,
          labels: sample.tokens.map(() => 'O'), // Initialize all as 'O'
          annotated: false,
        }));
        setAnnotations(initialAnnotations);
      }
    } catch (err) {
      console.error('Failed to load session:', err);
      setError('Failed to load annotation session');
      toast.error('Failed to load session');
    } finally {
      setLoading(false);
    }
  };

  const getCurrentSample = () => {
    return samples[currentSampleIndex];
  };

  const getCurrentAnnotation = () => {
    return annotations[currentSampleIndex];
  };

  const handleTokenClick = (tokenIndex) => {
    const newSelected = new Set(selectedTokens);
    if (newSelected.has(tokenIndex)) {
      newSelected.delete(tokenIndex);
    } else {
      newSelected.add(tokenIndex);
    }
    setSelectedTokens(newSelected);
  };

  const handleEntityTypeSelect = (entityType) => {
    setSelectedEntityType(entityType);
  };

  const handleApplyLabel = () => {
    if (selectedTokens.size === 0) {
      toast.error('Please select tokens to label');
      return;
    }

    const currentAnnotation = getCurrentAnnotation();
    if (!currentAnnotation) return;

    const newLabels = [...currentAnnotation.labels];
    const selectedIndices = Array.from(selectedTokens).sort((a, b) => a - b);
    
    // Apply BIO tagging
    selectedIndices.forEach((tokenIndex, i) => {
      if (selectedEntityType === 'O') {
        newLabels[tokenIndex] = 'O';
      } else {
        newLabels[tokenIndex] = i === 0 ? `B-${selectedEntityType}` : `I-${selectedEntityType}`;
      }
    });

    // Update annotations
    const newAnnotations = [...annotations];
    newAnnotations[currentSampleIndex] = {
      ...currentAnnotation,
      labels: newLabels,
      annotated: true,
    };
    setAnnotations(newAnnotations);
    setSelectedTokens(new Set());
    
    toast.success('Labels applied successfully');
  };

  const handleClearLabels = () => {
    const currentAnnotation = getCurrentAnnotation();
    if (!currentAnnotation) return;

    const newAnnotations = [...annotations];
    newAnnotations[currentSampleIndex] = {
      ...currentAnnotation,
      labels: currentAnnotation.tokens.map(() => 'O'),
      annotated: false,
    };
    setAnnotations(newAnnotations);
    setSelectedTokens(new Set());
    
    toast.success('Labels cleared');
  };

  const handleNextSample = () => {
    if (currentSampleIndex < samples.length - 1) {
      setCurrentSampleIndex(currentSampleIndex + 1);
      setSelectedTokens(new Set());
    }
  };

  const handlePreviousSample = () => {
    if (currentSampleIndex > 0) {
      setCurrentSampleIndex(currentSampleIndex - 1);
      setSelectedTokens(new Set());
    }
  };

  const handleSubmitAnnotations = async () => {
    const annotatedSamples = annotations.filter(ann => ann.annotated);
    
    if (annotatedSamples.length === 0) {
      toast.error('Please annotate at least one sample');
      return;
    }

    setSubmitting(true);
    try {
      const annotationData = annotatedSamples.map(ann => ({
        id: ann.id,
        tokens: ann.tokens,
        labels: ann.labels,
      }));

      await nerAPI.submitAnnotations(sessionId, annotationData);
      
      toast.success(`Submitted ${annotatedSamples.length} annotations`);
      setCompletionDialog(true);
    } catch (error) {
      toast.error('Failed to submit annotations');
    } finally {
      setSubmitting(false);
    }
  };

  const renderToken = (token, index) => {
    const currentAnnotation = getCurrentAnnotation();
    const label = currentAnnotation?.labels[index] || 'O';
    const isSelected = selectedTokens.has(index);
    
    const entityType = label.startsWith('B-') || label.startsWith('I-') 
      ? label.substring(2) 
      : label;
    
    const entityInfo = entityTypes.find(et => et.label === entityType) || entityTypes[4]; // Default to 'O'
    
    return (
      <Chip
        key={index}
        label={token}
        onClick={() => handleTokenClick(index)}
        variant={isSelected ? "filled" : "outlined"}
        sx={{
          margin: '2px',
          cursor: 'pointer',
          backgroundColor: isSelected 
            ? 'primary.main' 
            : (label !== 'O' ? entityInfo.color : 'transparent'),
          borderColor: isSelected 
            ? 'primary.main' 
            : (label !== 'O' ? entityInfo.borderColor : 'grey.300'),
          color: isSelected ? 'white' : 'inherit',
          '&:hover': {
            backgroundColor: isSelected 
              ? 'primary.dark' 
              : 'action.hover',
          },
        }}
      />
    );
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

  if (!session || samples.length === 0) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          No Samples Available for Annotation
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          This session may not be ready for annotation yet.
        </Typography>
        <Button variant="contained" onClick={() => navigate('/sessions')}>
          Back to Sessions
        </Button>
      </Paper>
    );
  }

  const currentSample = getCurrentSample();
  const currentAnnotation = getCurrentAnnotation();
  const annotatedCount = annotations.filter(ann => ann.annotated).length;
  const progress = (currentSampleIndex + 1) / samples.length * 100;

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <AnnotateIcon sx={{ mr: 2, fontSize: 'inherit' }} />
          Annotation Session
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Session: {sessionId.substring(0, 8)}... | Sample {currentSampleIndex + 1} of {samples.length}
        </Typography>
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" gutterBottom>
            Progress: {Math.round(progress)}% ({annotatedCount} annotated)
          </Typography>
          <LinearProgress variant="determinate" value={progress} />
        </Box>
      </Paper>

      <Grid container spacing={3}>
        {/* Main Annotation Area */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Text to Annotate
            </Typography>
            
            <Box
              sx={{
                p: 2,
                border: '1px solid #ddd',
                borderRadius: 1,
                backgroundColor: '#fafafa',
                minHeight: '100px',
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'flex-start',
                lineHeight: 1.8,
              }}
            >
              {currentSample?.tokens.map((token, index) => renderToken(token, index))}
            </Box>

            <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="outlined"
                onClick={handlePreviousSample}
                disabled={currentSampleIndex === 0}
              >
                Previous
              </Button>
              
              <Button
                variant="outlined"
                onClick={handleNextSample}
                disabled={currentSampleIndex === samples.length - 1}
              >
                Next
              </Button>
              
              <Button
                variant="outlined"
                color="warning"
                onClick={handleClearLabels}
              >
                Clear Labels
              </Button>
            </Box>
          </Paper>

          {/* Current Labels Display */}
          {currentAnnotation && (
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Current Labels
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {currentAnnotation.tokens.map((token, index) => {
                  const label = currentAnnotation.labels[index];
                  return (
                    <Typography key={index} variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {token}/{label}
                    </Typography>
                  );
                })}
              </Box>
            </Paper>
          )}
        </Grid>

        {/* Entity Types Panel */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Entity Types
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Select tokens, choose an entity type, then apply labels.
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 3 }}>
              {entityTypes.map((entityType) => (
                <Button
                  key={entityType.label}
                  variant={selectedEntityType === entityType.label ? "contained" : "outlined"}
                  onClick={() => handleEntityTypeSelect(entityType.label)}
                  sx={{
                    justifyContent: 'flex-start',
                    backgroundColor: selectedEntityType === entityType.label 
                      ? 'primary.main' 
                      : entityType.color,
                    borderColor: entityType.borderColor,
                    color: selectedEntityType === entityType.label 
                      ? 'white' 
                      : 'inherit',
                    '&:hover': {
                      backgroundColor: selectedEntityType === entityType.label 
                        ? 'primary.dark' 
                        : entityType.color,
                    },
                  }}
                >
                  <strong>{entityType.label}</strong>&nbsp;- {entityType.description}
                </Button>
              ))}
            </Box>

            <Button
              variant="contained"
              fullWidth
              onClick={handleApplyLabel}
              disabled={selectedTokens.size === 0}
              sx={{ mb: 2 }}
            >
              Apply {selectedEntityType} Label
            </Button>

            <Typography variant="body2" color="text.secondary">
              Selected tokens: {selectedTokens.size}
            </Typography>
          </Paper>

          {/* Submit Panel */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Submit Annotations
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Annotated: {annotatedCount} / {samples.length} samples
            </Typography>
            
            <Button
              variant="contained"
              fullWidth
              onClick={handleSubmitAnnotations}
              disabled={submitting || annotatedCount === 0}
              startIcon={submitting ? <CircularProgress size={20} /> : <SaveIcon />}
              sx={{ mb: 2 }}
            >
              {submitting ? 'Submitting...' : 'Submit Annotations'}
            </Button>

            <Button
              variant="outlined"
              fullWidth
              onClick={() => navigate('/sessions')}
            >
              Back to Sessions
            </Button>
          </Paper>
        </Grid>
      </Grid>

      {/* Completion Dialog */}
      <Dialog open={completionDialog} onClose={() => setCompletionDialog(false)}>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
          <CompleteIcon sx={{ mr: 1, color: 'success.main' }} />
          Annotations Submitted
        </DialogTitle>
        <DialogContent>
          <Typography>
            Your annotations have been submitted successfully! The model will be retrained with this new data.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => navigate('/sessions')}>
            Back to Sessions
          </Button>
          <Button 
            variant="contained" 
            onClick={() => navigate(`/results/${sessionId}`)}
          >
            View Results
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AnnotatePage;