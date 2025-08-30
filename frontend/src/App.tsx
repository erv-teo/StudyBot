import { Container, Stack, Title, Text, SimpleGrid } from '@mantine/core';
import { StatusCard } from './components/StatusCard';
import { ConfigCard } from './components/ConfigCard';
import { DocumentsTable } from './components/DocumentsTable';

export default function App() {
  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <Stack gap="xs">
          <Title order={1}>StudyBot Dashboard</Title>
          <Text c="dimmed" size="lg">
            Manage your RAG system documents and chunks
          </Text>
        </Stack>

        <SimpleGrid cols={{ base: 1, lg: 3 }} spacing="lg">
          <div style={{ gridColumn: 'span 2' }}>
            <StatusCard />
          </div>
          <div>
            <ConfigCard />
          </div>
        </SimpleGrid>

        <DocumentsTable />
      </Stack>
    </Container>
  );
}