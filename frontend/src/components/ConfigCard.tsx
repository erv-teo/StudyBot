import { Card, Text, Group, Stack, SimpleGrid, Skeleton, Alert, Box } from '@mantine/core';
import { IconSettings, IconAlertCircle } from '@tabler/icons-react';
import { useState, useEffect } from 'react';
import { apiClient, ConfigResponse } from '../lib/api';

export function ConfigCard() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        setError(null);
        const configData = await apiClient.getConfiguration();
        setConfig(configData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load configuration');
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  if (loading) {
    return (
      <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
        <Group>
          <IconSettings size={20} />
          <Stack gap={0}>
            <Text fw={500} size="lg">Bot Configuration</Text>
            <Text size="sm" c="dimmed">Current system configuration and settings</Text>
          </Stack>
        </Group>

        <Stack gap="md" mt="md">
          <SimpleGrid cols={2} spacing="md">
            {Array.from({ length: 6 }).map((_, i) => (
              <Stack key={i} gap={4}>
                <Skeleton height={16} width="60%" />
                <Skeleton height={20} width="80%" />
              </Stack>
            ))}
          </SimpleGrid>
        </Stack>
      </Card>
    );
  }

  if (error) {
    return (
      <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
        <Group>
          <IconSettings size={20} />
          <Stack gap={0}>
            <Text fw={500} size="lg">Bot Configuration</Text>
            <Text size="sm" c="dimmed">Current system configuration and settings</Text>
          </Stack>
        </Group>

        <Alert icon={<IconAlertCircle size={16} />} title="Error" color="red" mt="md">
          {error}
        </Alert>
      </Card>
    );
  }

  if (!config) {
    return (
      <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
        <Group>
          <IconSettings size={20} />
          <Stack gap={0}>
            <Text fw={500} size="lg">Bot Configuration</Text>
            <Text size="sm" c="dimmed">Current system configuration and settings</Text>
          </Stack>
        </Group>

        <Alert icon={<IconAlertCircle size={16} />} title="No Configuration" color="yellow" mt="md">
          No configuration data available.
        </Alert>
      </Card>
    );
  }

  return (
    <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
      <Group>
        <IconSettings size={20} />
        <Stack gap={0}>
          <Text fw={500} size="lg">Bot Configuration</Text>
          <Text size="sm" c="dimmed">Current system configuration and settings</Text>
        </Stack>
      </Group>

      <Stack gap="md" mt="md" justify="space-between">
        <SimpleGrid cols={2} spacing="md">
          <Stack gap={4}>
            <Text size="sm" fw={500} c="dimmed">Embedding Provider:</Text>
            <Text size="sm" style={{ textTransform: 'capitalize' }}>
              {config.embedding.provider} ({config.embedding.model_name})
            </Text>
          </Stack>
          <Stack gap={4}>
            <Text size="sm" fw={500} c="dimmed">LLM Provider:</Text>
            <Text size="sm" style={{ textTransform: 'capitalize' }}>
              {config.llm.provider} ({config.llm.model_name})
            </Text>
          </Stack>
          <Stack gap={4}>
            <Text size="sm" fw={500} c="dimmed">Vector Store:</Text>
            <Text size="sm" style={{ textTransform: 'capitalize' }}>
              {config.vectorstore.provider} ({config.vectorstore.collection_name})
            </Text>
          </Stack>
          <Stack gap={4}>
            <Text size="sm" fw={500} c="dimmed">Chunk Size:</Text>
            <Text size="sm">
              {config.chunking.chunk_size} chars (overlap: {config.chunking.chunk_overlap})
            </Text>
          </Stack>
          <Stack gap={4}>
            <Text size="sm" fw={500} c="dimmed">Retrieval:</Text>
            <Text size="sm">
              {config.rag.retrieval_k} documents
              {config.rag.score_threshold && ` (min score: ${config.rag.score_threshold})`}
            </Text>
          </Stack>
          <Stack gap={4}>
            <Text size="sm" fw={500} c="dimmed">Temperature:</Text>
            <Text size="sm">
              {config.llm.temperature}
              {config.llm.max_tokens && ` (max tokens: ${config.llm.max_tokens})`}
            </Text>
          </Stack>
        </SimpleGrid>
        
        <Text size="xs" c="dimmed" style={{ borderTop: '1px solid #e9ecef', paddingTop: 8}}>
          Log Level: {config.system.log_level} | Data Directory: {config.system.data_directory}
        </Text>
      </Stack>
    </Card>
  );
}