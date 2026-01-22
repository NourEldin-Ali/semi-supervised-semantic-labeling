import { useState } from 'react';
import TableCard from '../components/TableCard';
import FileUpload from '../components/FileUpload';
import Button from '../components/Button';
import Input from '../components/Input';
import Select from '../components/Select';
import RunStatistics from '../components/RunStatistics';
import {
  embeddingsApi,
  clusteringApi,
  labelingApi,
  classificationApi,
  downloadFile,
  EmbeddingResponse,
  ClusteringResponse,
  LabelingResponse,
  KNNTrainingResponse,
} from '../services/api';
import { EmbeddingConfig, ClusteringConfig, LabelingConfig, KNNConfig } from '../types';
import FullPipelineTab from '../components/FullPipelineTab';
import { Download } from 'lucide-react';

export default function WorkflowPage() {
  const [activeTab, setActiveTab] = useState<'step-by-step' | 'automated'>('step-by-step');
  
  const [steps, setSteps] = useState<{
    embeddings: { status: 'pending' | 'in-progress' | 'completed' | 'error'; data: EmbeddingResponse | null; error: string | undefined };
    clustering: { status: 'pending' | 'in-progress' | 'completed' | 'error'; data: ClusteringResponse | null; error: string | undefined };
    labeling: { status: 'pending' | 'in-progress' | 'completed' | 'error'; data: LabelingResponse | null; error: string | undefined };
    knn: { status: 'pending' | 'in-progress' | 'completed' | 'error'; data: KNNTrainingResponse | null; error: string | undefined };
  }>({
    embeddings: { status: 'pending' as const, data: null, error: undefined },
    clustering: { status: 'pending' as const, data: null, error: undefined },
    labeling: { status: 'pending' as const, data: null, error: undefined },
    knn: { status: 'pending' as const, data: null, error: undefined },
  });

  // Step 1: Embeddings
  const [csvFile1, setCsvFile1] = useState<File | null>(null);
  const [embeddingConfig, setEmbeddingConfig] = useState<EmbeddingConfig>({
    embedding_model: 'text-embedding-3-large',
    embed_type: 'open_ai',
    text_column: 'text',
    batch_size: 32,
  });

  // Step 2: Clustering
  const [csvFile2, setCsvFile2] = useState<File | null>(null);
  const [embeddingsFile, setEmbeddingsFile] = useState<File | null>(null);
  const [clusteringConfig, setClusteringConfig] = useState<ClusteringConfig>({
    k: -1, // -1 means auto-calculate
    metric: 'cosine',
    text_column: 'text',
  });

  // Step 3: Labeling
  const [clusteredCsvFile, setClusteredCsvFile] = useState<File | null>(null);
  const [originalCsvFile, setOriginalCsvFile] = useState<File | null>(null);
  const [labelingConfig, setLabelingConfig] = useState<LabelingConfig>({
    id_column: 'id',
    cluster_column: 'cluster_group',
    text_column: 'text',
    llm_model: 'gpt-5.2-2025-12-11',
    llm_type: 'open_ai',
    api_key: '',
  });

  // Step 4: KNN Training
  const [csvFile4, setCsvFile4] = useState<File | null>(null);
  const [embeddingsFile2, setEmbeddingsFile2] = useState<File | null>(null);
  const [knnConfig, setKnnConfig] = useState<KNNConfig>({
    id_column: 'id',
    label_column: 'labels',
  });

  const handleGenerateEmbeddings = async () => {
    if (!csvFile1) return;
    
    setSteps(prev => ({ ...prev, embeddings: { ...prev.embeddings, status: 'in-progress' } }));
    
    try {
      const formData = new FormData();
      formData.append('file', csvFile1);
      formData.append('embedding_model', embeddingConfig.embedding_model);
      formData.append('embed_type', embeddingConfig.embed_type);
      formData.append('batch_size', embeddingConfig.batch_size.toString());
      formData.append('text_column', embeddingConfig.text_column?.trim() || 'text');
      if (embeddingConfig.api_key) formData.append('api_key', embeddingConfig.api_key);

      const response = await embeddingsApi.generate(formData);
      setSteps(prev => ({ 
        ...prev, 
        embeddings: { status: 'completed', data: response, error: undefined } 
      }));
    } catch (error: any) {
      setSteps(prev => ({ 
        ...prev, 
        embeddings: { status: 'error', data: null, error: error.message } 
      }));
    }
  };

  const handleGenerateClusters = async () => {
    if (!csvFile2 || !embeddingsFile) return;
    
    setSteps(prev => ({ ...prev, clustering: { ...prev.clustering, status: 'in-progress' } }));
    
    try {
      const formData = new FormData();
      formData.append('csv_file', csvFile2);
      formData.append('embeddings_file', embeddingsFile);
      // k: -1 means auto-calculate, otherwise use provided value
      if (clusteringConfig.k !== -1) {
        formData.append('k', clusteringConfig.k.toString());
      }
      formData.append('metric', clusteringConfig.metric);
      formData.append('text_column', clusteringConfig.text_column?.trim() || 'text');

      const response = await clusteringApi.generate(formData);
      setSteps(prev => ({ 
        ...prev, 
        clustering: { status: 'completed', data: response, error: undefined } 
      }));
    } catch (error: any) {
      setSteps(prev => ({ 
        ...prev, 
        clustering: { status: 'error', data: null, error: error.message } 
      }));
    }
  };

  const handleGenerateLabels = async () => {
    if (!clusteredCsvFile || !originalCsvFile) return;
    
    setSteps(prev => ({ ...prev, labeling: { ...prev.labeling, status: 'in-progress' } }));
    
    try {
      const formData = new FormData();
      formData.append('clustered_csv_file', clusteredCsvFile);
      formData.append('original_csv_file', originalCsvFile);
      formData.append('id_column', labelingConfig.id_column);
      formData.append('cluster_column', labelingConfig.cluster_column);
      formData.append('llm_model', labelingConfig.llm_model);
      formData.append('llm_type', labelingConfig.llm_type);
      formData.append('text_column', labelingConfig.text_column?.trim() || 'text');
      if (labelingConfig.api_key) formData.append('api_key', labelingConfig.api_key);

      const response = await labelingApi.generateGroupLabels(formData);
      setSteps(prev => ({ 
        ...prev, 
        labeling: { status: 'completed', data: response, error: undefined } 
      }));
    } catch (error: any) {
      setSteps(prev => ({ 
        ...prev, 
        labeling: { status: 'error', data: null, error: error.message } 
      }));
    }
  };

  const handleTrainKNN = async () => {
    if (!csvFile4 || !embeddingsFile2) return;
    
    setSteps(prev => ({ ...prev, knn: { ...prev.knn, status: 'in-progress' } }));
    
    try {
      const formData = new FormData();
      formData.append('csv_file', csvFile4);
      formData.append('embeddings_file', embeddingsFile2);
      formData.append('id_column', knnConfig.id_column);
      formData.append('label_column', knnConfig.label_column);

      const response = await classificationApi.trainKNN(formData);
      setSteps(prev => ({ 
        ...prev, 
        knn: { status: 'completed', data: response, error: undefined } 
      }));
    } catch (error: any) {
      setSteps(prev => ({ 
        ...prev, 
        knn: { status: 'error', data: null, error: error.message } 
      }));
    }
  };

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Main Workflow</h1>
        <p className="mt-2 text-gray-600">Complete pipeline: Embeddings → Clustering → Labeling → KNN Training</p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('step-by-step')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'step-by-step'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Step-by-Step Workflow
          </button>
          <button
            onClick={() => setActiveTab('automated')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'automated'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Automated Full Pipeline
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'step-by-step' ? (
        <StepByStepWorkflow
          steps={steps}
          csvFile1={csvFile1}
          setCsvFile1={setCsvFile1}
          embeddingConfig={embeddingConfig}
          setEmbeddingConfig={setEmbeddingConfig}
          handleGenerateEmbeddings={handleGenerateEmbeddings}
          csvFile2={csvFile2}
          setCsvFile2={setCsvFile2}
          embeddingsFile={embeddingsFile}
          setEmbeddingsFile={setEmbeddingsFile}
          clusteringConfig={clusteringConfig}
          setClusteringConfig={setClusteringConfig}
          handleGenerateClusters={handleGenerateClusters}
          clusteredCsvFile={clusteredCsvFile}
          setClusteredCsvFile={setClusteredCsvFile}
          originalCsvFile={originalCsvFile}
          setOriginalCsvFile={setOriginalCsvFile}
          labelingConfig={labelingConfig}
          setLabelingConfig={setLabelingConfig}
          handleGenerateLabels={handleGenerateLabels}
          csvFile4={csvFile4}
          setCsvFile4={setCsvFile4}
          embeddingsFile2={embeddingsFile2}
          setEmbeddingsFile2={setEmbeddingsFile2}
          knnConfig={knnConfig}
          setKnnConfig={setKnnConfig}
          handleTrainKNN={handleTrainKNN}
        />
      ) : (
        <FullPipelineTab />
      )}
    </div>
  );
}

// Step-by-step workflow component
function StepByStepWorkflow(props: any) {
  return (
    <div className="space-y-6">
      {/* Step 1: Embeddings */}
      <TableCard
        title="Step 1: Generate Embeddings"
        step={1}
        status={props.steps.embeddings.status}
        error={props.steps.embeddings.error}
      >
        <div className="space-y-4">
          <FileUpload
            accept=".csv"
            file={props.csvFile1}
            onChange={props.setCsvFile1}
            label="Upload CSV File"
          />
          <Input
            label="Embedding Model"
            value={props.embeddingConfig.embedding_model}
            onChange={(e) => props.setEmbeddingConfig({ ...props.embeddingConfig, embedding_model: e.target.value })}
            placeholder="text-embedding-3-large"
          />
          <div className="grid grid-cols-3 gap-4">
            <Select
              label="Embedding Type"
              options={[
                { value: 'open_ai', label: 'OpenAI' },
                { value: 'ollama', label: 'Ollama' },
                { value: 'huggingface', label: 'HuggingFace' },
              ]}
              value={props.embeddingConfig.embed_type}
              onChange={(e) => props.setEmbeddingConfig({ ...props.embeddingConfig, embed_type: e.target.value as any })}
            />
            <Input
              label="Text Column"
              value={props.embeddingConfig.text_column || 'text'}
              onChange={(e) => props.setEmbeddingConfig({ ...props.embeddingConfig, text_column: e.target.value || 'text' })}
              placeholder="text"
            />
            <Input
              label="Batch Size"
              type="number"
              value={props.embeddingConfig.batch_size}
              onChange={(e) => props.setEmbeddingConfig({ ...props.embeddingConfig, batch_size: parseInt(e.target.value) })}
            />
          </div>
          <Button 
            onClick={props.handleGenerateEmbeddings} 
            disabled={!props.csvFile1 || props.steps.embeddings.status === 'in-progress'}
            loading={props.steps.embeddings.status === 'in-progress'}
          >
            Generate Embeddings
          </Button>
          {props.steps.embeddings.data && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg space-y-3">
              <p className="text-sm text-green-800">Embeddings generated successfully!</p>
              <p className="text-xs text-green-600">
                Shape: {props.steps.embeddings.data.embedding_shape?.join(' × ')}
              </p>
              <RunStatistics stats={props.steps.embeddings.data} />
              <div className="flex gap-2">
                <Button
                  onClick={() => downloadFile(props.steps.embeddings.data!.embeddings_file, `embeddings_${Date.now()}.npy`)}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Embeddings
                </Button>
                <Button
                  onClick={() => downloadFile(props.steps.embeddings.data!.csv_file, `original_${Date.now()}.csv`)}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download CSV
                </Button>
              </div>
            </div>
          )}
        </div>
      </TableCard>

      {/* Step 2: Clustering */}
      <TableCard
        title="Step 2: Generate Clusters"
        step={2}
        status={props.steps.clustering.status}
        error={props.steps.clustering.error}
      >
        <div className="space-y-4">
          <FileUpload
            accept=".csv"
            file={props.csvFile2}
            onChange={props.setCsvFile2}
            label="Upload Original CSV File"
          />
          <FileUpload
            accept=".npy"
            file={props.embeddingsFile}
            onChange={props.setEmbeddingsFile}
            label="Upload Embeddings File (.npy)"
          />
          <div className="grid grid-cols-3 gap-4">
            <Input
              label="Number of Clusters (k: -1 by default to be automatically calculated)"
              type="number"
              value={props.clusteringConfig.k}
              onChange={(e) => props.setClusteringConfig({ ...props.clusteringConfig, k: parseInt(e.target.value) || -1 })}
            />
            <Input
              label="Text Column"
              value={props.clusteringConfig.text_column || 'text'}
              onChange={(e) => props.setClusteringConfig({ ...props.clusteringConfig, text_column: e.target.value || 'text' })}
              placeholder="text"
            />
            <Input
              label="Metric"
              value={props.clusteringConfig.metric}
              onChange={(e) => props.setClusteringConfig({ ...props.clusteringConfig, metric: e.target.value })}
            />
          </div>
          <Button 
            onClick={props.handleGenerateClusters} 
            disabled={!props.csvFile2 || !props.embeddingsFile || props.steps.clustering.status === 'in-progress'}
            loading={props.steps.clustering.status === 'in-progress'}
          >
            Generate Clusters
          </Button>
          {props.steps.clustering.data && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg space-y-3">
              <p className="text-sm text-green-800">
                Found {props.steps.clustering.data.number_of_clusters} clusters!
              </p>
              <RunStatistics stats={props.steps.clustering.data} />
              <div>
                <Button
                  onClick={() => downloadFile(props.steps.clustering.data!.output_csv_file, `clustered_${Date.now()}.csv`)}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Clustered CSV
                </Button>
              </div>
            </div>
          )}
        </div>
      </TableCard>

      {/* Step 3: Labeling */}
      <TableCard
        title="Step 3: Generate Labels"
        step={3}
        status={props.steps.labeling.status}
        error={props.steps.labeling.error}
      >
        <div className="space-y-4">
          <FileUpload
            accept=".csv"
            file={props.clusteredCsvFile}
            onChange={props.setClusteredCsvFile}
            label="Upload Clustered CSV File (from Step 2)"
            disabled={props.steps.labeling.status === 'in-progress'}
          />
          <FileUpload
            accept=".csv"
            file={props.originalCsvFile}
            onChange={props.setOriginalCsvFile}
            label="Upload Original CSV File"
            disabled={props.steps.labeling.status === 'in-progress'}
          />
          <p className="text-sm text-gray-500">
            Clusters will be automatically extracted from the clustered CSV file. Items can belong to multiple clusters (comma-separated cluster IDs).
          </p>
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="ID Column"
              value={props.labelingConfig.id_column}
              onChange={(e) => props.setLabelingConfig({ ...props.labelingConfig, id_column: e.target.value })}
              disabled={props.steps.labeling.status === 'in-progress'}
            />
            <Input
              label="Cluster Column (in clustered CSV)"
              value={props.labelingConfig.cluster_column}
              onChange={(e) => props.setLabelingConfig({ ...props.labelingConfig, cluster_column: e.target.value })}
              disabled={props.steps.labeling.status === 'in-progress'}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Text Column"
              value={props.labelingConfig.text_column || 'text'}
              onChange={(e) => props.setLabelingConfig({ ...props.labelingConfig, text_column: e.target.value || 'text' })}
              placeholder="text"
              disabled={props.steps.labeling.status === 'in-progress'}
            />
            <Input
              label="LLM Model"
              value={props.labelingConfig.llm_model}
              onChange={(e) => props.setLabelingConfig({ ...props.labelingConfig, llm_model: e.target.value })}
              placeholder="gpt-5.2-2025-12-11"
              disabled={props.steps.labeling.status === 'in-progress'}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Select
              label="LLM Type"
              value={props.labelingConfig.llm_type}
              onChange={(e) => props.setLabelingConfig({ ...props.labelingConfig, llm_type: e.target.value as any })}
              options={[
                { value: 'open_ai', label: 'OpenAI' },
                { value: 'groq_ai', label: 'Groq AI' },
                { value: 'ollama', label: 'Ollama' },
                { value: 'anthropic', label: 'Anthropic' },
              ]}
              disabled={props.steps.labeling.status === 'in-progress'}
            />
            <Input
              label="API Key (optional)"
              type="password"
              value={props.labelingConfig.api_key || ''}
              onChange={(e) => props.setLabelingConfig({ ...props.labelingConfig, api_key: e.target.value || undefined })}
              placeholder="Leave empty to use environment variable"
              disabled={props.steps.labeling.status === 'in-progress'}
            />
          </div>
          <Button 
            onClick={props.handleGenerateLabels} 
            disabled={!props.clusteredCsvFile || !props.originalCsvFile || props.steps.labeling.status === 'in-progress'}
            loading={props.steps.labeling.status === 'in-progress'}
          >
            Generate Labels
          </Button>
          {props.steps.labeling.data && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg space-y-3">
              <p className="text-sm text-green-800">
                {props.steps.labeling.data.labeled_items_count} items labeled!
              </p>
              <RunStatistics stats={props.steps.labeling.data} />
              <div>
                <Button
                  onClick={() => downloadFile(props.steps.labeling.data!.output_csv_file, `labeled_${Date.now()}.csv`)}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Labeled CSV
                </Button>
              </div>
            </div>
          )}
        </div>
      </TableCard>

      {/* Step 4: KNN Training */}
      <TableCard
        title="Step 4: Train KNN Model"
        step={4}
        status={props.steps.knn.status}
        error={props.steps.knn.error}
      >
        <div className="space-y-4">
          <FileUpload
            accept=".csv"
            file={props.csvFile4}
            onChange={props.setCsvFile4}
            label="Upload Labeled CSV File"
          />
          <FileUpload
            accept=".npy"
            file={props.embeddingsFile2}
            onChange={props.setEmbeddingsFile2}
            label="Upload Embeddings File (.npy)"
          />
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="ID Column"
              value={props.knnConfig.id_column}
              onChange={(e) => props.setKnnConfig({ ...props.knnConfig, id_column: e.target.value })}
            />
            <Input
              label="Label Column"
              value={props.knnConfig.label_column}
              onChange={(e) => props.setKnnConfig({ ...props.knnConfig, label_column: e.target.value })}
            />
          </div>
          <Button 
            onClick={props.handleTrainKNN} 
            disabled={!props.csvFile4 || !props.embeddingsFile2 || props.steps.knn.status === 'in-progress'}
            loading={props.steps.knn.status === 'in-progress'}
          >
            Train KNN Model
          </Button>
          {props.steps.knn.data && (
            <div className="mt-4 p-4 bg-green-50 rounded-lg space-y-3">
              <p className="text-sm text-green-800">
                Model trained with {props.steps.knn.data.training_samples} samples!
              </p>
              <RunStatistics stats={props.steps.knn.data} />
              <div className="flex gap-2">
                <Button
                  onClick={() => downloadFile(props.steps.knn.data!.model_file, `knn_model_${Date.now()}.joblib`)}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download KNN Model
                </Button>
                <Button
                  onClick={() => downloadFile(props.steps.knn.data!.csv_file, `training_data_${Date.now()}.csv`)}
                  variant="secondary"
                  className="flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download CSV
                </Button>
              </div>
            </div>
          )}
        </div>
      </TableCard>
    </div>
  );
}
