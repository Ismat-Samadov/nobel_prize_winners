import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Container, AppBar, Toolbar, Typography, Box } from '@mui/material';
import HomePage from './pages/HomePage';
import PredictPage from './pages/PredictPage';
import TrainPage from './pages/TrainPage';
import SessionsPage from './pages/SessionsPage';
import AnnotatePage from './pages/AnnotatePage';
import ResultsPage from './pages/ResultsPage';
import Navigation from './components/Navigation';

function App() {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Active Learning NER
          </Typography>
        </Toolbar>
      </AppBar>
      
      <Navigation />
      
      <Container maxWidth="lg" sx={{ mt: 3, mb: 3 }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/annotate/:sessionId" element={<AnnotatePage />} />
          <Route path="/results/:sessionId" element={<ResultsPage />} />
        </Routes>
      </Container>
    </Box>
  );
}

export default App;