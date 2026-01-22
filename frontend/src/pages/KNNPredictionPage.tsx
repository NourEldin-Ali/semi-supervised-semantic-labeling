import { useState } from 'react';
import FileUpload from '../components/FileUpload';
import Button from '../components/Button';
import Input from '../components/Input';
import RunStatistics from '../components/RunStatistics';
import { classificationApi, downloadFile } from '../services/api';
import { Download } from 'lucide-react';

export default function KNNPredictionPage() {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [embeddingsFile, setEmbeddingsFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [idColumn, setIdColumn] = useState('id');
  const [k, setK] = useState(3);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!csvFile || !embeddingsFile || !modelFile) {
      setError('Please upload all required files');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('csv_file', csvFile);
      formData.append('embeddings_file', embeddingsFile);
      formData.append('model_file', modelFile);
      formData.append('id_column', idColumn);
      formData.append('k', k.toString());

      const response = await classificationApi.predict(formData);
      setResult(response);
    } catch (err: any) {
      setError(err.message || 'Failed to generate predictions');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (result?.output_csv_file) {
      downloadFile(result.output_csv_file, `predictions_${Date.now()}.csv`);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">KNN Prediction</h1>
        <p className="mt-2 text-gray-600">Use a trained KNN model to predict labels for new data</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <FileUpload
          accept=".csv"
          file={csvFile}
          onChange={setCsvFile}
          label="Upload CSV File with New Data"
          disabled={loading}
        />

        <FileUpload
          accept=".npy"
          file={embeddingsFile}
          onChange={setEmbeddingsFile}
          label="Upload Embeddings File (.npy)"
          disabled={loading}
        />

        <FileUpload
          accept=".joblib"
          file={modelFile}
          onChange={setModelFile}
          label="Upload Trained KNN Model File (.joblib)"
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
            label="Number of Neighbors (k)"
            type="number"
            value={k}
            onChange={(e) => setK(parseInt(e.target.value))}
            disabled={loading}
          />
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <Button onClick={handlePredict} loading={loading} disabled={!csvFile || !embeddingsFile || !modelFile || loading}>
          Generate Predictions
        </Button>

        {result && (
          <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-lg space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-green-900">Predictions Generated!</h3>
                <p className="text-sm text-green-700 mt-1">
                  {result.predicted_items_count} items predicted out of {result.total_items} total
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
