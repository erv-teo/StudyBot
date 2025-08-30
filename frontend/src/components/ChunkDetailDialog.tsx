import { useState } from 'react';
import { Modal, Button, Textarea, Text, Group, Stack, SimpleGrid, Alert, Code } from '@mantine/core';
import { Chunk, ChunkUpdateRequest } from '../lib/api';
import { apiClient } from '../lib/api';
import { IconEdit, IconDeviceFloppy, IconX, IconAlertCircle } from '@tabler/icons-react';

interface ChunkDetailDialogProps {
  chunk: Chunk | null;
  opened: boolean;
  onClose: () => void;
  onChunkUpdated: () => void;
}

export function ChunkDetailDialog({ chunk, opened, onClose, onChunkUpdated }: ChunkDetailDialogProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset form when chunk changes
  useState(() => {
    if (chunk) {
      setContent(chunk.content);
      setIsEditing(false);
      setError(null);
    }
  });

  const handleSave = async () => {
    if (!chunk) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const updateData: ChunkUpdateRequest = { content };
      await apiClient.updateChunk(chunk.chunk_id, updateData);
      setIsEditing(false);
      onChunkUpdated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update chunk');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!chunk) return;
    
    if (!confirm('Are you sure you want to delete this chunk?')) return;
    
    setLoading(true);
    setError(null);
    
    try {
      await apiClient.deleteChunk(chunk.chunk_id);
      onClose();
      onChunkUpdated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete chunk');
    } finally {
      setLoading(false);
    }
  };

  if (!chunk) return null;

  return (
    <Modal 
      opened={opened} 
      onClose={onClose} 
      title={
        <Group justify="space-between" w="100%">
          <Text fw={500}>Chunk Details</Text>
          <Group>
            {!isEditing ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsEditing(true)}
                disabled={loading}
                leftSection={<IconEdit size={16} />}
              >
                Edit
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsEditing(false)}
                disabled={loading}
                leftSection={<IconX size={16} />}
              >
                Cancel
              </Button>
            )}
          </Group>
        </Group>
      }
      size="xl"
    >
      <Stack gap="md">
        {error && (
          <Alert icon={<IconAlertCircle size={16} />} title="Error" color="red">
            {error}
          </Alert>
        )}

        <div>
          <Text fw={500} mb="xs">Content</Text>
          {isEditing ? (
            <Textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Enter chunk content..."
              minRows={8}
              disabled={loading}
            />
          ) : (
            <Code block style={{ maxHeight: '300px', overflow: 'auto' }}>
              {chunk.content}
            </Code>
          )}
        </div>

        <div>
          <Text fw={500} mb="xs">Metadata</Text>
          <Code block style={{ maxHeight: '200px', overflow: 'auto' }}>
            {JSON.stringify(chunk.metadata, null, 2)}
          </Code>
        </div>

        <SimpleGrid cols={2} spacing="md">
          <div>
            <Text size="sm" fw={500} c="dimmed">Content Length:</Text>
            <Text size="sm">{chunk.content_length} characters</Text>
          </div>
          <div>
            <Text size="sm" fw={500} c="dimmed">Created:</Text>
            <Text size="sm">
              {chunk.created_at ? new Date(chunk.created_at).toLocaleString() : 'Unknown'}
            </Text>
          </div>
        </SimpleGrid>

        <Group justify="flex-end" mt="md">
          <Button
            variant="outline"
            color="red"
            onClick={handleDelete}
            disabled={loading}
          >
            Delete Chunk
          </Button>
          {isEditing && (
            <Button
              onClick={handleSave}
              disabled={loading}
              leftSection={<IconDeviceFloppy size={16} />}
            >
              Save Changes
            </Button>
          )}
        </Group>
      </Stack>
    </Modal>
  );
}
