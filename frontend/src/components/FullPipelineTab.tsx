import { useState } from 'react';
import FileUpload from './FileUpload';
import Button from './Button';
import Input from './Input';
import Select from './Select';
import RunStatistics from './RunStatistics';
import { workflowApi, downloadFile } from '../services/api';
import { Download, CheckCircle2, Loader2 } from 'lucide-react';

export default function FullPipelineTab() {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  // Embedding configuration
  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-3-large');
  const [embedType, setEmbedType] = useState('open_ai');
  const [textColumn, setTextColumn] = useState('text');
  const [batchSize, setBatchSize] = useState(32);
  const [embeddingApiKey, setEmbeddingApiKey] = useState('');

  // Clustering configuration
  const [k, setK] = useState(-1); // -1 means auto-calculate
  const [metric, setMetric] = useState('cosine');
  const [clusteringTextColumn, setClusteringTextColumn] = useState('text');

  // Labeling configuration
  const [idColumn, setIdColumn] = useState('id');
  const [llmModel, setLlmModel] = useState('gpt-5.2-2025-12-11');
  const [llmType, setLlmType] = useState('open_ai');
  const [llmApiKey, setLlmApiKey] = useState('');

  // KNN configuration
  const [labelColumn, setLabelColumn] = useState('labels');

  const handleExecutePipeline = async () => {
    if (!csvFile) {
      setError('Please upload a CSV file');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('csv_file', csvFile);
      
      // Embedding config
      formData.append('embedding_model', embeddingModel);
      formData.append('embed_type', embedType);
      formData.append('batch_size', batchSize.toString());
      formData.append('text_column', (textColumn?.trim() || clusteringTextColumn?.trim() || 'text'));
      if (embeddingApiKey) formData.append('embedding_api_key', embeddingApiKey);
      
      // Clustering config
      if (k !== -1) {
        formData.append('k', k.toString());
      }
      formData.append('metric', metric);
      
      // Labeling config
      formData.append('id_column', idColumn);
      formData.append('llm_model', llmModel);
      formData.append('llm_type', llmType);
      if (llmApiKey) formData.append('llm_api_key', llmApiKey);
      
      // KNN config
      formData.append('label_column', labelColumn);

      const response = await workflowApi.executeFullPipeline(formData);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Failed to execute pipeline');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadLabeledCSV = () => {
    if (result?.labeled_csv_file) {
      downloadFile(result.labeled_csv_file, `labeled_${Date.now()}.csv`);
    }
  };

  const handleDownloadKNNModel = () => {
    if (result?.knn_model_file) {
      downloadFile(result.knn_model_file, `knn_model_${Date.now()}.joblib`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-800">
          <strong>Automated Pipeline:</strong> Upload a CSV file and configure all settings. 
          The system will automatically execute Embeddings → Clustering → Labeling → KNN Training 
          and return both the labeled CSV and trained KNN model.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        {/* File Upload */}
        <FileUpload
          accept=".csv"
          file={csvFile}
          onChange={setCsvFile}
          label="Upload CSV File"
          disabled={loading}
        />

        {/* Embedding Configuration */}
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-4">Embedding Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Embedding Model"
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              placeholder="text-embedding-3-large"
              disabled={loading}
            />
            <Select
              label="Embedding Type"
              options={[
                { value: 'open_ai', label: 'OpenAI' },
                { value: 'ollama', label: 'Ollama' },
                { value: 'huggingface', label: 'HuggingFace' },
              ]}
              value={embedType}
              onChange={(e) => setEmbedType(e.target.value)}
              disabled={loading}
            />
            <Input
              label="Text Column"
              value={textColumn}
              onChange={(e) => setTextColumn(e.target.value)}
              placeholder="text"
              disabled={loading}
            />
            <Input
              label="Batch Size"
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              disabled={loading}
            />
            <Input
              label="Embedding API Key (optional)"
              type="password"
              value={embeddingApiKey}
              onChange={(e) => setEmbeddingApiKey(e.target.value)}
              placeholder="Leave empty to use .env"
              disabled={loading}
            />
          </div>
        </div>

        {/* Clustering Configuration */}
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-4">Clustering Configuration</h3>
          <div className="grid grid-cols-3 gap-4">
            <Input
              label="Number of Clusters (k: -1 by default to be automatically calculated)"
              type="number"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value) || -1)}
              disabled={loading}
            />
            <Input
              label="Text Column"
              value={clusteringTextColumn}
              onChange={(e) => setClusteringTextColumn(e.target.value)}
              placeholder="text"
              disabled={loading}
            />
            <Input
              label="Metric"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              disabled={loading}
            />
          </div>
        </div>

        {/* Labeling Configuration */}
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-4">Labeling Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="ID Column"
              value={idColumn}
              onChange={(e) => setIdColumn(e.target.value)}
              placeholder="id"
              disabled={loading}
            />
            <Input
              label="LLM Model"
              value={llmModel}
              onChange={(e) => setLlmModel(e.target.value)}
              placeholder="gpt-5.2-2025-12-11"
              disabled={loading}
            />
            <Select
              label="LLM Type"
              options={[
                { value: 'open_ai', label: 'OpenAI' },
                { value: 'groq_ai', label: 'Groq AI' },
                { value: 'ollama', label: 'Ollama' },
                { value: 'anthropic', label: 'Anthropic' },
              ]}
              value={llmType}
              onChange={(e) => setLlmType(e.target.value)}
              disabled={loading}
            />
            <Input
              label="LLM API Key (optional)"
              type="password"
              value={llmApiKey}
              onChange={(e) => setLlmApiKey(e.target.value)}
              placeholder="Leave empty to use .env"
              disabled={loading}
            />
          </div>
        </div>

        {/* KNN Configuration */}
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-4">KNN Training Configuration</h3>
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Label Column"
              value={labelColumn}
              onChange={(e) => setLabelColumn(e.target.value)}
              placeholder="labels"
              disabled={loading}
            />
          </div>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <Button 
          onClick={handleExecutePipeline} 
          loading={loading} 
          disabled={!csvFile || loading} 
          className="w-full"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Executing Pipeline...</span>
            </>
          ) : (
            'Execute Full Pipeline'
          )}
        </Button>

        {result && (
          <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-2 mb-4">
              <CheckCircle2 className="h-6 w-6 text-green-600" />
              <h3 className="text-lg font-semibold text-green-900">Pipeline Completed Successfully!</h3>
            </div>

            {/* Statistics */}
            <div className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Total Items" value={result.statistics.total_items} />
              <StatCard label="Labeled Items" value={result.statistics.labeled_items} />
              <StatCard label="Clusters" value={result.statistics.number_of_clusters} />
              <StatCard label="Training Samples" value={result.statistics.training_samples} />
            </div>
            <RunStatistics stats={result.statistics} className="mb-6" />

            {/* Download Buttons */}
            <div className="flex space-x-4">
              <Button variant="secondary" onClick={handleDownloadLabeledCSV}>
                <Download className="h-4 w-4" />
                <span>Download Labeled CSV</span>
              </Button>
              <Button variant="secondary" onClick={handleDownloadKNNModel}>
                <Download className="h-4 w-4" />
                <span>Download KNN Model</span>
              </Button>
            </div>

            <div className="mt-4 text-sm text-green-700">
              <p>Embedding Shape: {result.statistics.embedding_shape.join(' × ')}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-white p-3 rounded-lg border border-green-200">
      <p className="text-xs text-gray-600">{label}</p>
      <p className="text-xl font-bold text-gray-900">{value}</p>
    </div>
  );
}
