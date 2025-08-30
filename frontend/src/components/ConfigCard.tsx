import { Card, Text, Group, Stack, SimpleGrid } from '@mantine/core';
import { IconSettings } from '@tabler/icons-react';

export function ConfigCard() {
  return (
    <Card shadow="sm" padding="lg" radius="md" withBorder>
      <Group>
        <IconSettings size={20} />
        <div>
          <Text fw={500} size="lg">Bot Configuration</Text>
          <Text size="sm" c="dimmed">Current system configuration and settings</Text>
        </div>
      </Group>

      <Stack gap="md" mt="md">
        <SimpleGrid cols={2} spacing="md">
          <div>
            <Text size="sm" fw={500} c="dimmed">Embedding Provider:</Text>
            <Text size="sm" mt={4}>OpenAI</Text>
          </div>
          <div>
            <Text size="sm" fw={500} c="dimmed">LLM Provider:</Text>
            <Text size="sm" mt={4}>OpenAI GPT-4</Text>
          </div>
          <div>
            <Text size="sm" fw={500} c="dimmed">Vector Store:</Text>
            <Text size="sm" mt={4}>ChromaDB</Text>
          </div>
          <div>
            <Text size="sm" fw={500} c="dimmed">Chunk Size:</Text>
            <Text size="sm" mt={4}>1000 tokens</Text>
          </div>
        </SimpleGrid>
        
        <Text size="xs" c="dimmed" style={{ borderTop: '1px solid #e9ecef', paddingTop: 8 }}>
          Configuration details will be available once the API endpoint is implemented.
        </Text>
      </Stack>
    </Card>
  );
}
