import { useState, useEffect, useCallback } from 'react';
import { apiClient, Document, Chunk, HealthStatus } from '../lib/api';

export function useHealthStatus() {
  const [status, setStatus] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getHealthStatus();
      setStatus(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch health status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    // Poll every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  return { status, loading, error, refetch: fetchStatus };
}

export function useDocuments() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getDocuments();
      setDocuments(data.documents);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch documents');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  return { documents, loading, error, refetch: fetchDocuments };
}

export function useDocumentChunks(source: string) {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchChunks = useCallback(async () => {
    if (!source) return;
    
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getDocumentChunks(source);
      setChunks(data.chunks);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch chunks');
    } finally {
      setLoading(false);
    }
  }, [source]);

  useEffect(() => {
    fetchChunks();
  }, [fetchChunks]);

  return { chunks, loading, error, refetch: fetchChunks };
}

export function useChunk(chunkId: string) {
  const [chunk, setChunk] = useState<Chunk | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchChunk = useCallback(async () => {
    if (!chunkId) return;
    
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getChunk(chunkId);
      setChunk(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch chunk');
    } finally {
      setLoading(false);
    }
  }, [chunkId]);

  useEffect(() => {
    fetchChunk();
  }, [fetchChunk]);

  return { chunk, loading, error, refetch: fetchChunk };
}
