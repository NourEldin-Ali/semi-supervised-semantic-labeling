export interface WorkflowStep {
  id: string;
  name: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  data?: any;
  error?: string;
}

export interface FileUpload {
  file: File;
  name: string;
  type: 'csv' | 'embeddings' | 'model';
}

export interface EmbeddingConfig {
  embedding_model: string;
  embed_type: 'open_ai' | 'ollama' | 'huggingface';
  text_column: string;
  batch_size: number;
  api_key?: string;
}

export interface ClusteringConfig {
  k: number; // -1 means auto-calculated as (number of items - 1) on backend
  metric: string;
  text_column: string;
}

export interface LabelingConfig {
  id_column: string;
  cluster_column: string;
  text_column: string;
  llm_model: string;
  llm_type: 'open_ai' | 'groq_ai' | 'ollama' | 'anthropic';
  api_key?: string;
}

export interface KNNConfig {
  id_column: string;
  label_column: string;
  k?: number;
}

export interface EvaluationConfig {
  id_column: string;
  label_column: string;
}
