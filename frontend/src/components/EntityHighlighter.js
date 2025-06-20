import React from 'react';
import { Box, Chip, Typography } from '@mui/material';

const entityColors = {
  'PER': '#e3f2fd',    // Light blue for persons
  'ORG': '#f3e5f5',    // Light purple for organizations
  'LOC': '#e8f5e8',    // Light green for locations
  'MISC': '#fff3e0',   // Light orange for miscellaneous
};

const entityBorderColors = {
  'PER': '#1976d2',    // Blue
  'ORG': '#7b1fa2',    // Purple
  'LOC': '#388e3c',    // Green
  'MISC': '#f57c00',   // Orange
};

const EntityHighlighter = ({ tokens, labels, entities }) => {
  if (!tokens || !labels) {
    return <Typography>No data to display</Typography>;
  }

  const renderTokens = () => {
    const result = [];
    let i = 0;

    while (i < tokens.length) {
      const token = tokens[i];
      const label = labels[i];

      if (label && label.startsWith('B-')) {
        // Start of entity
        const entityType = label.substring(2);
        let entityText = token;
        let j = i + 1;

        // Collect all I- tokens for this entity
        while (j < tokens.length && labels[j] === `I-${entityType}`) {
          entityText += ' ' + tokens[j];
          j++;
        }

        result.push(
          <Chip
            key={i}
            label={entityText}
            size="small"
            variant="outlined"
            sx={{
              backgroundColor: entityColors[entityType] || '#f5f5f5',
              borderColor: entityBorderColors[entityType] || '#ccc',
              margin: '2px',
              '& .MuiChip-label': {
                fontSize: '0.875rem',
              },
            }}
          />
        );

        i = j; // Move to next token after entity
      } else if (label === 'O' || !label.startsWith('I-')) {
        // Outside entity or single token
        result.push(
          <span key={i} style={{ margin: '2px 4px' }}>
            {token}
          </span>
        );
        i++;
      } else {
        // This shouldn't happen with proper BIO tagging
        i++;
      }
    }

    return result;
  };

  const renderEntitiesList = () => {
    if (!entities || entities.length === 0) {
      return null;
    }

    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Detected Entities:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {entities.map((entity, index) => (
            <Chip
              key={index}
              label={`${entity.text} (${entity.label})`}
              variant="filled"
              sx={{
                backgroundColor: entityColors[entity.label] || '#f5f5f5',
                color: entityBorderColors[entity.label] || '#000',
                fontWeight: 'medium',
              }}
            />
          ))}
        </Box>
      </Box>
    );
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Annotated Text:
      </Typography>
      <Box
        sx={{
          p: 2,
          border: '1px solid #ddd',
          borderRadius: 1,
          backgroundColor: '#fafafa',
          minHeight: '60px',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'flex-start',
          lineHeight: 1.8,
        }}
      >
        {renderTokens()}
      </Box>
      {renderEntitiesList()}
      
      {/* Legend */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Entity Types:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {Object.entries(entityColors).map(([type, color]) => (
            <Chip
              key={type}
              label={type}
              size="small"
              sx={{
                backgroundColor: color,
                borderColor: entityBorderColors[type],
                border: `1px solid ${entityBorderColors[type]}`,
              }}
            />
          ))}
        </Box>
      </Box>
    </Box>
  );
};

export default EntityHighlighter;