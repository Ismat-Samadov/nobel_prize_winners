import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Tabs,
  Tab,
  Paper,
} from '@mui/material';
import {
  Home as HomeIcon,
  Psychology as PredictIcon,
  School as TrainIcon,
  Dashboard as SessionsIcon,
} from '@mui/icons-material';

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const tabs = [
    { label: 'Home', value: '/', icon: <HomeIcon /> },
    { label: 'Predict', value: '/predict', icon: <PredictIcon /> },
    { label: 'Train', value: '/train', icon: <TrainIcon /> },
    { label: 'Sessions', value: '/sessions', icon: <SessionsIcon /> },
  ];

  const currentTab = tabs.find(tab => tab.value === location.pathname)?.value || '/';

  const handleTabChange = (event, newValue) => {
    navigate(newValue);
  };

  return (
    <Paper sx={{ mt: 2, mx: 2 }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={currentTab}
          onChange={handleTabChange}
          aria-label="navigation tabs"
          variant="fullWidth"
        >
          {tabs.map((tab) => (
            <Tab
              key={tab.value}
              label={tab.label}
              value={tab.value}
              icon={tab.icon}
              iconPosition="start"
            />
          ))}
        </Tabs>
      </Box>
    </Paper>
  );
};

export default Navigation;