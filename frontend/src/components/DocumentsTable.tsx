
import { useState } from 'react';
import { 
  Table, 
  Button, 
  Text, 
  Group, 
  Stack, 
  Card, 
  Collapse, 
  ActionIcon,
  Alert,
  LoadingOverlay,
  Badge
} from '@mantine/core';
import { useDocuments, useDocumentChunks } from '../hooks/useApi';
import { ChunkDetailDialog } from './ChunkDetailDialog';
import { Chunk } from '../lib/api';
import { 
  IconFileText, 
  IconChevronDown, 
  IconEye, 
  IconRefresh,
  IconAlertCircle 
} from '@tabler/icons-react';

export function DocumentsTable() {
  const { documents, loading, error, refetch } = useDocuments();
  const [selectedChunk, setSelectedChunk] = useState<Chunk | null>(null);
  const [dialogOpened, setDialogOpened] = useState(false);
  const [expandedDocument, setExpandedDocument] = useState<string | null>(null);

  const handleChunkClick = (chunk: Chunk) => {
    setSelectedChunk(chunk);
    setDialogOpened(true);
  };

  const handleChunkUpdated = () => {
    refetch();
  };

  const toggleDocument = (source: string) => {
    setExpandedDocument(expandedDocument === source ? null : source);
  };

  if (loading) {
    return (
      <Card shadow="sm" padding="lg" radius="md" withBorder pos="relative">
        <LoadingOverlay visible />
        <div style={{ height: 200 }} />
      </Card>
    );
  }

  if (error) {
    return (
      <Alert icon={<IconAlertCircle size={16} />} title="Error" color="red">
        <Text mb="md">{error}</Text>
        <Button variant="outline" size="sm" onClick={refetch}>
          Retry
        </Button>
      </Alert>
    );
  }

  if (documents.length === 0) {
    return (
      <Card shadow="sm" padding="xl" radius="md" withBorder>
        <Stack align="center" gap="md">
          <IconFileText size={48} color="#adb5bd" />
          <Text size="lg" fw={500}>No documents found</Text>
          <Text c="dimmed">Upload some documents to get started.</Text>
        </Stack>
      </Card>
    );
  }

  return (
    <Stack gap="md">
      <Group justify="space-between">
        <Text size="xl" fw={600}>Documents & Chunks</Text>
        <Button variant="outline" size="sm" onClick={refetch} leftSection={<IconRefresh size={16} />}>
          Refresh
        </Button>
      </Group>

      <Stack gap="xs">
        {documents.map((document) => (
          <DocumentCard
            key={document.source}
            document={document}
            isExpanded={expandedDocument === document.source}
            onToggle={() => toggleDocument(document.source)}
            onChunkClick={handleChunkClick}
          />
        ))}
      </Stack>

      <ChunkDetailDialog
        chunk={selectedChunk}
        opened={dialogOpened}
        onClose={() => setDialogOpened(false)}
        onChunkUpdated={handleChunkUpdated}
      />
    </Stack>
  );
}

interface DocumentCardProps {
  document: {
    doc_id: string;
    source: string;
    original_filename?: string;
    chunks_count: number;
    total_chars: number;
    content_type: string;
    created_at: string;
  };
  isExpanded: boolean;
  onToggle: () => void;
  onChunkClick: (chunk: Chunk) => void;
}

function DocumentCard({ document, isExpanded, onToggle, onChunkClick }: DocumentCardProps) {
  const { chunks, loading, error } = useDocumentChunks(document.doc_id);

  return (
    <Card shadow="sm" radius="md" withBorder>
      <Group justify="space-between" onClick={onToggle} style={{ cursor: 'pointer' }}>
        <Group>
          <IconFileText size={20} color="#6c757d" />
          <div>
            <Text fw={500} style={{ maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {document.original_filename || document.source}
            </Text>
            <Text size="sm" c="dimmed">
              {document.chunks_count} chunks • {document.total_chars.toLocaleString()} chars
            </Text>
          </div>
        </Group>
        <IconChevronDown 
          size={20} 
          style={{ 
            transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s'
          }} 
        />
      </Group>

      <Collapse in={isExpanded}>
        <Stack gap="md" mt="md">
          {loading ? (
            <LoadingOverlay visible />
          ) : error ? (
            <Alert icon={<IconAlertCircle size={16} />} title="Error" color="red">
              Error loading chunks: {error}
            </Alert>
          ) : chunks.length === 0 ? (
            <Text size="sm" c="dimmed" ta="center" py="md">
              No chunks found for this document.
            </Text>
          ) : (
            <Table>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Preview</Table.Th>
                  <Table.Th>Length</Table.Th>
                  <Table.Th>Created</Table.Th>
                  <Table.Th style={{ width: 100 }}>Actions</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {chunks.map((chunk) => (
                  <Table.Tr key={chunk.chunk_id}>
                    <Table.Td>
                      <Text size="sm" style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {chunk.content_preview || chunk.content.substring(0, 150)}...
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Badge variant="light" size="sm">
                        {chunk.content_length} chars
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Text size="sm" c="dimmed">
                        {chunk.created_at ? new Date(chunk.created_at).toLocaleDateString() : 'Unknown'}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <ActionIcon
                        variant="subtle"
                        size="sm"
                        onClick={() => onChunkClick(chunk)}
                      >
                        <IconEye size={16} />
                      </ActionIcon>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          )}
        </Stack>
      </Collapse>
    </Card>
  );
}
