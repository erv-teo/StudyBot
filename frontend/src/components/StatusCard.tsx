import { Card, Text, Group, Badge, Stack, Flex, Box } from '@mantine/core';
import { useHealthStatus } from '../hooks/useApi';
import { IconActivity, IconCheck, IconX, IconAlertCircle } from '@tabler/icons-react';

export function StatusCard() {
  const { status, loading, error } = useHealthStatus();

  if (loading) {
    return (
      <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
        <Group>
          <IconActivity size={20} style={{ animation: 'spin 1s linear infinite' }} />
          <Stack gap={0}>
            <Text fw={500} size="lg">Bot Status</Text>
            <Text size="sm" c="dimmed">Checking system health...</Text>
          </Stack>
        </Group>
        <Stack gap="xs" mt="md">
          <Box h={16} bg="#f1f3f5" style={{ borderRadius: 4 }} />
          <Box h={16} bg="#f1f3f5" style={{ borderRadius: 4, width: '75%' }} />
        </Stack>
      </Card>
    );
  }

  if (error) {
    return (
      <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
        <Group>
          <IconX size={20} color="red" />
          <Stack gap={0}>
            <Text fw={500} size="lg" c="red">Bot Status</Text>
            <Text size="sm" c="dimmed">Unable to connect to bot</Text>
          </Stack>
        </Group>
        <Text size="sm" c="red" mt="md">{error}</Text>
      </Card>
    );
  }

  const isHealthy = status?.status === 'healthy';
  const services = status?.services;

  return (
    <Card shadow="sm" padding="lg" radius="md" withBorder h="100%">
      <Group>
        {isHealthy ? (
          <IconCheck size={20} color="green" />
        ) : (
          <IconAlertCircle size={20} color="orange" />
        )}
        <Stack gap={0}>
          <Text fw={500} size="lg">Bot Status</Text>
          <Text size="sm" c="dimmed">
            {isHealthy ? 'All systems operational' : 'Some services may be affected'}
          </Text>
        </Stack>
      </Group>

      <Stack gap="md" mt="md">
        <Flex justify="space-between" align="center">
          <Text size="sm" fw={500}>Overall Status</Text>
          <Badge color={isHealthy ? 'green' : 'yellow'}>
            {status?.status.toUpperCase()}
          </Badge>
        </Flex>
        
        <Stack gap="xs">
          <Flex justify="space-between" align="center">
            <Text size="sm">RAG Pipeline</Text>
            {services?.rag_pipeline ? (
              <IconCheck size={16} color="green" />
            ) : (
              <IconX size={16} color="red" />
            )}
          </Flex>
          <Flex justify="space-between" align="center">
            <Text size="sm">Chunk Manager</Text>
            {services?.chunk_manager ? (
              <IconCheck size={16} color="green" />
            ) : (
              <IconX size={16} color="red" />
            )}
          </Flex>
          <Flex justify="space-between" align="center">
            <Text size="sm">Vector Store</Text>
            {services?.vector_store ? (
              <IconCheck size={16} color="green" />
            ) : (
              <IconX size={16} color="red" />
            )}
          </Flex>
        </Stack>
        
        {status?.timestamp && (
          <Text size="xs" c="dimmed" style={{ borderTop: '1px solid #e9ecef', paddingTop: 8 }}>
            Last updated: {new Date(status.timestamp).toLocaleString()}
          </Text>
        )}
      </Stack>
    </Card>
  );
}