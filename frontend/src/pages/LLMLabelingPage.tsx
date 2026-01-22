import { useState } from 'react';
import FileUpload from '../components/FileUpload';
import Button from '../components/Button';
import Input from '../components/Input';
import Select from '../components/Select';
import RunStatistics from '../components/RunStatistics';
import { labelingApi, downloadFile } from '../services/api';
import { Download } from 'lucide-react';

export default function LLMLabelingPage() {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [idColumn, setIdColumn] = useState('id');
  const [textColumn, setTextColumn] = useState('text');
  const [llmModel, setLlmModel] = useState('gpt-5.2-2025-12-11');
  const [llmType, setLlmType] = useState('open_ai');
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateLabels = async () => {
    if (!csvFile) {
      setError('Please upload a CSV file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('csv_file', csvFile);
      formData.append('id_column', idColumn);
      formData.append('llm_model', llmModel);
      formData.append('llm_type', llmType);
      formData.append('text_column', textColumn?.trim() || 'text');
      if (apiKey) formData.append('api_key', apiKey);

      const response = await labelingApi.generateItemLabels(formData);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Failed to generate labels');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (result?.output_csv_file) {
      downloadFile(result.output_csv_file, `labeled_${Date.now()}.csv`);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">LLM Labeling</h1>
        <p className="mt-2 text-gray-600">Use LLM to generate labels for each item in your CSV file</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <FileUpload
          accept=".csv"
          file={csvFile}
          onChange={setCsvFile}
          label="Upload CSV File"
          disabled={loading}
        />

        <div className="grid grid-cols-2 gap-4">
          <Input
            label="ID Column"
            value={idColumn}
            onChange={(e) => setIdColumn(e.target.value)}
            placeholder="id"
            disabled={loading}
          />
          <Input
            label="Text Column"
            value={textColumn}
            onChange={(e) => setTextColumn(e.target.value)}
            placeholder="text"
            disabled={loading}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
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
        </div>

        <Input
          label="API Key (optional, can use .env)"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Leave empty to use .env"
          disabled={loading}
        />

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <Button onClick={handleGenerateLabels} loading={loading} disabled={!csvFile || loading}>
          Generate Labels
        </Button>

        {result && (
          <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-lg space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-green-900">Labels Generated!</h3>
                <p className="text-sm text-green-700 mt-1">
                  {result.labeled_items_count} items labeled out of {result.total_items} total
                </p>
              </div>
              <Button variant="secondary" onClick={handleDownload}>
                <Download className="h-4 w-4" />
                <span>Download CSV</span>
              </Button>
            </div>
            <RunStatistics stats={result} />
            <div className="text-sm text-green-600">
              <p>Output file: {result.output_csv_file}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
