import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

/** Run statistics returned by all execution endpoints. */
export interface RunStats {
  execution_time_seconds?: number;
  tokens_consumed?: number;
  energy_consumed_kwh?: number;
  emissions_kg_co2eq?: number;
}

export interface EmbeddingResponse extends RunStats {
  message: string;
  embeddings_file: string;
  csv_file: string;
  embedding_shape: number[];
  text_column: string;
}

export interface ClusteringResponse extends RunStats {
  message: string;
  output_csv_file: string;
  original_csv_file: string;
  number_of_clusters: number;
  cluster_summary: Array<{
    cluster_id: number;
    item_count: number;
    item_indices: number[];
  }>;
}

export interface LabelingResponse extends RunStats {
  message: string;
  output_csv_file: string;
  original_csv_file: string;
  labeled_items_count: number;
  total_items: number;
  labels_generated?: number;
}

export interface KNNTrainingResponse extends RunStats {
  message: string;
  model_file: string;
  csv_file: string;
  embeddings_file: string;
  training_samples: number;
  total_samples: number;
}

export interface KNNPredictionResponse extends RunStats {
  message: string;
  output_csv_file: string;
  original_csv_file: string;
  predicted_items_count: number;
  total_items: number;
}

export interface EvaluationResponse {
  message: string;
  file1: string;
  file2: string;
  evaluation_results: {
    total_questions_evaluated: number;
    method1_name: string;
    method2_name: string;
    ignored_data?: {
      ids_only_in_file1: number;
      ids_only_in_file2: number;
      skipped_no_labels: number;
      skipped_no_text: number;
    };
    execution_time_seconds?: number;
    tokens_consumed?: number;
    energy_consumed_kwh?: number;
    emissions_kg_co2eq?: number;
    average_metrics: {
      method1: {
        intent_alignment_score: number;
        concept_completeness_score: number;
        noise_redundancy_penalty: number;
        terminology_normalization_score: number;
        audit_usefulness_score: number;
        control_mapping_clarity_score: number;
      };
      method2: {
        intent_alignment_score: number;
        concept_completeness_score: number;
        noise_redundancy_penalty: number;
        terminology_normalization_score: number;
        audit_usefulness_score: number;
        control_mapping_clarity_score: number;
      };
    };
    question_metrics: Array<{
      id: string;
      method1: {
        method: string;
        intent_alignment_score: number;
        concept_completeness_score: number;
        noise_redundancy_penalty: number;
        terminology_normalization_score: number;
        audit_usefulness_score: number;
        control_mapping_clarity_score: number;
        reasoning: string;
        labels: string[];
      };
      method2: {
        method: string;
        intent_alignment_score: number;
        concept_completeness_score: number;
        noise_redundancy_penalty: number;
        terminology_normalization_score: number;
        audit_usefulness_score: number;
        control_mapping_clarity_score: number;
        reasoning: string;
        labels: string[];
      };
    }>;
  };
}

export interface PaperEvaluationResponse {
  message: string;
  file1: string;
  file2: string;
  evaluation_results: {
    method1_name: string;
    method2_name: string;
    total_questions_evaluated: number;
    pairwise_summary?: {
      method1_name: string;
      method2_name: string;
      wins: number;
      losses: number;
      ties: number;
      win_rate_pct: number;
      loss_rate_pct: number;
      tie_rate_pct: number;
      total_pairs: number;
    };
    average_scores?: {
      method1: Record<string, { mean: number; std: number }>;
      method2: Record<string, { mean: number; std: number }>;
    };
    dimension_breakdown?: Array<{
      dimension: string;
      method1_mean: number;
      method2_mean: number;
      delta: number;
    }>;
    latex_table?: string;
    statistical_test?: {
      test: string;
      wins: number;
      losses: number;
      ties: number;
      n: number;
      p_value: number | null;
      alpha: number;
      significant: boolean;
      interpretation: string;
    };
    ignored_data: {
      ids_only_in_file1: number;
      ids_only_in_file2: number;
      skipped_no_labels: number;
      skipped_no_text: number;
    };
    files?: {
      pairwise_judgments_csv?: string | null;
      absolute_scores_csv?: string | null;
      pairwise_summary_csv?: string | null;
      average_scores_csv?: string | null;
      dimension_breakdown_csv?: string | null;
      json?: string | null;
    };
    random_seed: number;
    run_pairwise?: boolean;
    run_absolute?: boolean;
    execution_time_seconds?: number;
    tokens_consumed?: number;
    energy_consumed_kwh?: number;
    emissions_kg_co2eq?: number;
  };
}

export interface SelectQuestionResult {
  id: string;
  question: string;
  score?: number | null;
  labels?: string[];
  matched_labels?: string[];
}

export interface SelectQuestionResponse extends RunStats {
  message: string;
  method: 'bm25' | 'embedding' | 'label_embedding';
  results: SelectQuestionResult[];
  total_results: number;
}

// Embeddings API
export const embeddingsApi = {
  generate: async (formData: FormData): Promise<EmbeddingResponse> => {
    const response = await api.post<EmbeddingResponse>('/embeddings/generate', formData);
    return response.data;
  },
};

// Clustering API
export const clusteringApi = {
  generate: async (formData: FormData): Promise<ClusteringResponse> => {
    const response = await api.post<ClusteringResponse>('/clustering/generate', formData);
    return response.data;
  },
};

// Labeling API
export const labelingApi = {
  generateGroupLabels: async (formData: FormData): Promise<LabelingResponse> => {
    const response = await api.post<LabelingResponse>('/labeling/generate-group-labels', formData);
    return response.data;
  },
  generateItemLabels: async (formData: FormData): Promise<LabelingResponse> => {
    const response = await api.post<LabelingResponse>('/labeling/generate-item-labels', formData);
    return response.data;
  },
};

// Classification API
export const classificationApi = {
  trainKNN: async (formData: FormData): Promise<KNNTrainingResponse> => {
    const response = await api.post<KNNTrainingResponse>('/classification/train-knn', formData);
    return response.data;
  },
  predict: async (formData: FormData): Promise<KNNPredictionResponse> => {
    const response = await api.post<KNNPredictionResponse>('/classification/predict', formData);
    return response.data;
  },
};

// Evaluation API
export const evaluationApi = {
  compare: async (formData: FormData): Promise<EvaluationResponse> => {
    const response = await api.post<EvaluationResponse>('/evaluation/compare', formData);
    return response.data;
  },
  paper: async (formData: FormData): Promise<PaperEvaluationResponse> => {
    const response = await api.post<PaperEvaluationResponse>('/evaluation/paper', formData);
    return response.data;
  },
};

// Workflow API
export interface FullPipelineResponse {
  message: string;
  original_csv_file: string;
  labeled_csv_file: string;
  knn_model_file: string;
  statistics: {
    total_items: number;
    labeled_items: number;
    number_of_clusters: number;
    training_samples: number;
    embedding_shape: number[];
  } & RunStats;
}

export const workflowApi = {
  executeFullPipeline: async (formData: FormData): Promise<FullPipelineResponse> => {
    const response = await api.post<FullPipelineResponse>('/workflow/full-pipeline', formData);
    return response.data;
  },
};

// Select Question API
export const selectQuestionApi = {
  bm25: async (formData: FormData): Promise<SelectQuestionResponse> => {
    const response = await api.post<SelectQuestionResponse>('/select-question/bm25', formData);
    return response.data;
  },
  embedding: async (formData: FormData): Promise<SelectQuestionResponse> => {
    const response = await api.post<SelectQuestionResponse>('/select-question/embedding', formData);
    return response.data;
  },
  labelEmbedding: async (formData: FormData): Promise<SelectQuestionResponse> => {
    const response = await api.post<SelectQuestionResponse>('/select-question/label-embedding', formData);
    return response.data;
  },
};

// File download helper - determines the correct endpoint based on file extension
export const downloadFile = async (filePath: string, filename?: string) => {
  // Extract filename from path if not provided
  const actualFilename = filename || filePath.split('/').pop() || filePath.split('\\').pop() || 'download';
  
  // Encode the file path properly - handle both forward and backward slashes
  // FastAPI path parameters handle URL encoding, but we need to ensure proper encoding
  const encodedPath = encodeURIComponent(filePath.replace(/\\/g, '/'));
  
  // Determine endpoint based on file extension and path
  let endpoint = '';
  if (filePath.includes('outputs')) {
    endpoint = `/labeling/download/${encodedPath}`;
  } else if (filePath.endsWith('.npy')) {
    endpoint = `/embeddings/download/${encodedPath}`;
  } else if (filePath.endsWith('.joblib')) {
    endpoint = `/classification/download/${encodedPath}`;
  } else if (filePath.includes('clustered') || filePath.includes('clustering')) {
    endpoint = `/clustering/download/${encodedPath}`;
  } else if (filePath.includes('labeled') || filePath.includes('labeling')) {
    endpoint = `/labeling/download/${encodedPath}`;
  } else if (filePath.endsWith('.csv')) {
    // For CSV files, try to determine the correct endpoint based on path
    if (filePath.includes('outputs')) {
      endpoint = `/labeling/download/${encodedPath}`;
    } else {
      endpoint = `/classification/download/${encodedPath}`;
    }
  } else {
    // Fallback: try embeddings endpoint
    endpoint = `/embeddings/download/${encodedPath}`;
  }
  
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to download file: ${response.statusText} - ${errorText}`);
    }
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = actualFilename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  } catch (error) {
    console.error('Download error:', error);
    throw error;
  }
};

export default api;
