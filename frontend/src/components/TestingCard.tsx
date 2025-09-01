import { useState } from 'react';
import { Card, Title, Text, TextInput, Button, Stack, Paper, Loader, Alert } from '@mantine/core';
import { IconPlayerPlay, IconAlertCircle } from '@tabler/icons-react';
import { apiClient, RAGQueryResponse } from '../lib/api';

export function TestingCard() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<RAGQueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const disabled = loading || !query.trim();

  const handleQuery = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const result = await apiClient.queryRAG({ question: query.trim() });
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get response');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleQuery();
    }
  };

  return (
    <Card shadow="sm" padding="lg" radius="md" withBorder>
      <Stack gap="md">
        <Title order={3}>Testing</Title>
        
        {/* Query Input */}
        <Paper 
          p="xs" 
          bg="dark.4" 
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px',
            borderRadius: '8px'
          }}
        >
          <Button
            variant="subtle"
            color="gray"
            size="sm"
            onClick={handleQuery}
            disabled={disabled}
            style={{ 
              minWidth: '40px', 
              height: '40px',
              padding: '0',
              backgroundColor: 'var(--mantine-color-dark-3)',
              color: disabled ? 'var(--mantine-color-gray-5)' : 'var(--mantine-color-gray-1)'
            }}
          >
            {loading ? (
              <Loader size="sm" color="gray" />
            ) : (
              <IconPlayerPlay size={16}/>
            )}
          </Button>
          
          <TextInput
            placeholder="Ask a question..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            variant="unstyled"
            style={{ flex: 1 }}
            styles={{
              input: {
                backgroundColor: 'transparent',
                color: 'var(--mantine-color-gray-2)',
                fontSize: '14px',
                '&::placeholder': {
                  color: 'var(--mantine-color-gray-5)'
                }
              }
            }}
          />
        </Paper>

        {/* Error Display */}
        {error && (
          <Alert 
            icon={<IconAlertCircle size={16} />} 
            title="Error" 
            color="red"
            variant="light"
          >
            {error}
          </Alert>
        )}

        {/* Response Display */}
        {response && (
          <Paper p="md" bg="dark.3" radius="md">
            <Stack gap="xs">
              <Text size="sm" c="gray.4" fw={500}>Question:</Text>
              <Text size="sm" c="gray.2">{response.question}</Text>
              
              <Text size="sm" c="gray.4" fw={500} mt="md">Answer:</Text>
              <Text size="sm" c="gray.1" style={{ whiteSpace: 'pre-wrap' }}>
                {response.answer}
              </Text>
              
              {response.processing_time_ms && (
                <Text size="xs" c="gray.4" mt="xs">
                  Processed in {response.processing_time_ms.toFixed(0)}ms
                </Text>
              )}
            </Stack>
          </Paper>
        )}
      </Stack>
    </Card>
  );
}
