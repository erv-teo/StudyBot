const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Document {
  doc_id: string;
  source: string;
  original_filename?: string;
  chunks_count: number;
  total_chars: number;
  content_type: string;
  created_at: string;
}

export interface DocumentListResponse {
  documents: Document[];
  total_count: number;
}

export interface Chunk {
  chunk_id: string;
  content: string;
  content_preview?: string;
  metadata: Record<string, any>;
  content_length: number;
  created_at?: string;
}

export interface ChunkListResponse {
  chunks: Chunk[];
  total_count: number;
  limit: number;
  source_filter?: string;
}

export interface ChunkUpdateRequest {
  content?: string;
  metadata?: Record<string, any>;
}

export interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  version: string;
  services: {
    rag_pipeline: boolean;
    chunk_manager: boolean;
    vector_store: boolean;
  };
}

export interface ConfigResponse {
  embedding: {
    provider: string;
    model_name: string;
  };
  llm: {
    provider: string;
    model_name: string;
    temperature: number;
    max_tokens?: number;
  };
  vectorstore: {
    provider: string;
    collection_name: string;
  };
  chunking: {
    chunk_size: number;
    chunk_overlap: number;
  };
  rag: {
    retrieval_k: number;
    score_threshold?: number;
    enable_query_analysis: boolean;
  };
  system: {
    log_level: string;
    data_directory: string;
  };
}

export interface DeleteResponse {
  success: true;
  message: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

     // Don't try to parse JSON for 204 No Content responses
    if (response.status === 204) {
      return { success: true, message: 'Resource deleted successfully' } as T;
    }

    return response.json();
  }

  // Health check
  async getHealthStatus(): Promise<HealthStatus> {
    return this.request<HealthStatus>('/health');
  }

  // Configuration
  async getConfiguration(): Promise<ConfigResponse> {
    return this.request<ConfigResponse>('/config/');
  }

  // Documents
  async getDocuments(): Promise<DocumentListResponse> {
    return this.request<DocumentListResponse>('/documents/');
  }

  async getDocumentStats(source: string) {
    return this.request(`/documents/${encodeURIComponent(source)}/stats`);
  }

  async deleteDocument(source: string): Promise<DeleteResponse> {
    return this.request<DeleteResponse>(`/documents/${encodeURIComponent(source)}`, {
      method: 'DELETE',
    });
  }

  // Chunks
  async getChunks(sourceFilter?: string, limit: number = 50): Promise<ChunkListResponse> {
    const params = new URLSearchParams();
    if (sourceFilter) params.append('source_filter', sourceFilter);
    params.append('limit', limit.toString());
    
    return this.request<ChunkListResponse>(`/chunks/?${params.toString()}`);
  }

  async getChunk(chunkId: string): Promise<Chunk> {
    return this.request<Chunk>(`/chunks/${chunkId}`);
  }

  async updateChunk(chunkId: string, data: ChunkUpdateRequest): Promise<Chunk> {
    return this.request<Chunk>(`/chunks/${chunkId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteChunk(chunkId: string): Promise<DeleteResponse> {
    return this.request<DeleteResponse>(`/chunks/${chunkId}`, {
      method: 'DELETE',
    });
  }

  async getDocumentChunks(docId: string): Promise<ChunkListResponse> {
    return this.request<ChunkListResponse>(`/chunks/document/${docId}`);
  }
}

export const apiClient = new ApiClient();
