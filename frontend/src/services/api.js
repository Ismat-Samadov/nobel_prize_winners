import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth (if needed)
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const nerAPI = {
  // Prediction endpoints
  predict: async (text, modelId = 'default') => {
    const response = await api.post('/predict', { text, model_id: modelId });
    return response.data;
  },

  // File upload
  uploadFile: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Training endpoints
  startTraining: async (config) => {
    const response = await api.post('/train', config);
    return response.data;
  },

  // Session management
  getSessions: async () => {
    const response = await api.get('/sessions');
    return response.data;
  },

  getSession: async (sessionId) => {
    const response = await api.get(`/sessions/${sessionId}`);
    return response.data;
  },

  deleteSession: async (sessionId) => {
    const response = await api.delete(`/sessions/${sessionId}`);
    return response.data;
  },

  // Annotation
  submitAnnotations: async (sessionId, annotations) => {
    const response = await api.post('/annotate', {
      session_id: sessionId,
      annotations,
    });
    return response.data;
  },

  // Results
  getResults: async (sessionId) => {
    const response = await api.get(`/results/${sessionId}`);
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;