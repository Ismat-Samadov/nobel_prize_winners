import React, { useState, useEffect } from 'react';
import {
  Typography,
  Paper,
  Box,
  Button,
  CircularProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Refresh as RefreshIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
  Edit as AnnotateIcon,
  Assessment as ResultsIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { nerAPI } from '../services/api';
import toast from 'react-hot-toast';

const SessionsPage = () => {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteDialog, setDeleteDialog] = useState({ open: false, sessionId: null });

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await nerAPI.getSessions();
      setSessions(result.sessions || []);
    } catch (err) {
      console.error('Failed to load sessions:', err);
      setError('Failed to load training sessions');
      toast.error('Failed to load sessions');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    try {
      await nerAPI.deleteSession(sessionId);
      setSessions(prev => prev.filter(s => s.session_id !== sessionId));
      toast.success('Session deleted successfully');
      setDeleteDialog({ open: false, sessionId: null });
    } catch (error) {
      toast.error('Failed to delete session');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
      case 'active_learning':
        return 'primary';
      case 'error':
        return 'error';
      case 'initializing':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'active_learning':
        return 'Ready for Annotation';
      case 'initializing':
        return 'Initializing';
      case 'running':
        return 'Training';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Error';
      default:
        return status;
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const getProgressPercentage = (currentRound, totalRounds) => {
    return totalRounds > 0 ? (currentRound / totalRounds) * 100 : 0;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h4" sx={{ display: 'flex', alignItems: 'center' }}>
            <DashboardIcon sx={{ mr: 2, fontSize: 'inherit' }} />
            Training Sessions
          </Typography>
          <Box>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={loadSessions}
              sx={{ mr: 2 }}
            >
              Refresh
            </Button>
            <Button
              variant="contained"
              onClick={() => navigate('/train')}
            >
              New Session
            </Button>
          </Box>
        </Box>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
          Monitor and manage your active learning training sessions
        </Typography>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {sessions.length === 0 ? (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" gutterBottom>
            No Training Sessions Found
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Start your first active learning session to begin training a custom NER model.
          </Typography>
          <Button
            variant="contained"
            onClick={() => navigate('/train')}
          >
            Start New Session
          </Button>
        </Paper>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Session ID</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Progress</TableCell>
                <TableCell>Labeled Samples</TableCell>
                <TableCell>Unlabeled Samples</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Updated</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sessions.map((session) => (
                <TableRow key={session.session_id}>
                  <TableCell>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                      {session.session_id.substring(0, 8)}...
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={getStatusText(session.status)}
                      color={getStatusColor(session.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={getProgressPercentage(session.current_round, session.total_rounds)}
                        sx={{ width: 60, mr: 1 }}
                      />
                      <Typography variant="body2">
                        {session.current_round}/{session.total_rounds}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>{session.labeled_samples}</TableCell>
                  <TableCell>{session.unlabeled_samples}</TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {formatDate(session.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {formatDate(session.updated_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <IconButton
                        size="small"
                        onClick={() => navigate(`/sessions/${session.session_id}`)}
                        title="View Details"
                      >
                        <ViewIcon />
                      </IconButton>
                      
                      {session.status === 'active_learning' && (
                        <IconButton
                          size="small"
                          onClick={() => navigate(`/annotate/${session.session_id}`)}
                          title="Annotate"
                          color="primary"
                        >
                          <AnnotateIcon />
                        </IconButton>
                      )}
                      
                      {(session.status === 'completed' || session.current_round > 0) && (
                        <IconButton
                          size="small"
                          onClick={() => navigate(`/results/${session.session_id}`)}
                          title="View Results"
                          color="success"
                        >
                          <ResultsIcon />
                        </IconButton>
                      )}
                      
                      <IconButton
                        size="small"
                        onClick={() => setDeleteDialog({ open: true, sessionId: session.session_id })}
                        title="Delete"
                        color="error"
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialog.open}
        onClose={() => setDeleteDialog({ open: false, sessionId: null })}
      >
        <DialogTitle>Delete Training Session</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this training session? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setDeleteDialog({ open: false, sessionId: null })}
          >
            Cancel
          </Button>
          <Button
            onClick={() => handleDeleteSession(deleteDialog.sessionId)}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SessionsPage;