import { useState } from 'react';
import FileUpload from '../components/FileUpload';
import Button from '../components/Button';
import Input from '../components/Input';
import Select from '../components/Select';
import RunStatistics from '../components/RunStatistics';
import { evaluationApi } from '../services/api';

export default function EvaluationPage() {
  const [csvFile1, setCsvFile1] = useState<File | null>(null);
  const [csvFile2, setCsvFile2] = useState<File | null>(null);
  const [idColumn, setIdColumn] = useState('id');
  const [textColumn, setTextColumn] = useState('text');
  const [labelColumn, setLabelColumn] = useState('labels');
  const [method1Name, setMethod1Name] = useState('Method 1');
  const [method2Name, setMethod2Name] = useState('Method 2');
  const [llmModel, setLlmModel] = useState('gpt-5.2-2025-12-11');
  const [llmType, setLlmType] = useState('open_ai');
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!csvFile1 || !csvFile2) {
      setError('Please upload both CSV files');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('csv_file1', csvFile1);
      formData.append('csv_file2', csvFile2);
      formData.append('id_column', idColumn);
      formData.append('label_column', labelColumn);
      formData.append('method1_name', method1Name);
      formData.append('method2_name', method2Name);
      formData.append('llm_model', llmModel);
      formData.append('llm_type', llmType);
      formData.append('text_column', textColumn?.trim() || 'text');
      if (apiKey) formData.append('api_key', apiKey);

      const response = await evaluationApi.compare(formData);
      setResult(response.evaluation_results);
    } catch (err: any) {
      setError(err.message || 'Failed to evaluate files');
    } finally {
      setLoading(false);
    }
  };

  const escapeCsv = (v: string): string => {
    if (v == null || v === undefined) return '';
    const s = String(v);
    if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };

  const handleDownloadResults = () => {
    if (!result?.question_metrics) return;
    const m1 = result.method1_name ?? 'Method 1';
    const m2 = result.method2_name ?? 'Method 2';
    type PairwiseJudgeItem = {
      id: string;
      winner: 'method1' | 'method2' | 'tie';
      reasoning: string;
    };
    const pairwise: PairwiseJudgeItem[] = result.pairwise_judge?.per_question ?? [];
    const pairwiseById = new Map<string, PairwiseJudgeItem>(
      pairwise.map((p) => [String(p.id), p])
    );
    const headers = [
      'Question ID',
      `${m1} Correctness`, `${m1} Completeness`, `${m1} Generalization`, `${m1} Consistency`,
      `${m1} Reasoning`, `${m1} Labels`,
      `${m2} Correctness`, `${m2} Completeness`, `${m2} Generalization`, `${m2} Consistency`,
      `${m2} Reasoning`, `${m2} Labels`,
      'LLM Winner', 'LLM Winner Reasoning',
    ];
    const rows: string[] = [headers.map(escapeCsv).join(',')];
    for (const q of result.question_metrics) {
      const r1 = q.method1;
      const r2 = q.method2;
      const pw = pairwiseById.get(String(q.id));
      let winnerLabel = '';
      if (pw?.winner === 'method1') winnerLabel = m1;
      else if (pw?.winner === 'method2') winnerLabel = m2;
      else if (pw?.winner === 'tie') winnerLabel = 'Tie';
      rows.push([
        escapeCsv(q.id),
        r1.correctness, r1.completeness, r1.generalization, r1.consistency,
        escapeCsv(r1.reasoning ?? ''), escapeCsv((r1.labels ?? []).join(', ')),
        r2.correctness, r2.completeness, r2.generalization, r2.consistency,
        escapeCsv(r2.reasoning ?? ''), escapeCsv((r2.labels ?? []).join(', ')),
        escapeCsv(winnerLabel), escapeCsv(pw?.reasoning ?? ''),
      ].join(','));
    }
    const avg = result.average_metrics;
    if (avg?.method1 && avg?.method2) {
      rows.push('');
      rows.push([
        'AVERAGE',
        avg.method1.correctness.toFixed(2), avg.method1.completeness.toFixed(2),
        avg.method1.generalization.toFixed(2), avg.method1.consistency.toFixed(2),
        '', '',
        avg.method2.correctness.toFixed(2), avg.method2.completeness.toFixed(2),
        avg.method2.generalization.toFixed(2), avg.method2.consistency.toFixed(2),
        '', '',
        '', '',
      ].join(','));
    }
    const csv = '\uFEFF' + rows.join('\r\n'); // BOM for Excel UTF-8
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `evaluation-results-${m1.replace(/\W/g, '-')}-vs-${m2.replace(/\W/g, '-')}-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">LLM-Based Evaluation</h1>
        <p className="mt-2 text-gray-600">
          Use LLM as a judge to evaluate and compare two labeling methods with detailed metrics per question
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div className="grid grid-cols-2 gap-6">
          <FileUpload
            accept=".csv"
            file={csvFile1}
            onChange={setCsvFile1}
            label="Method 1 CSV File"
            disabled={loading}
          />
          <FileUpload
            accept=".csv"
            file={csvFile2}
            onChange={setCsvFile2}
            label="Method 2 CSV File"
            disabled={loading}
          />
        </div>

        <div className="grid grid-cols-3 gap-4">
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
          <Input
            label="Label Column"
            value={labelColumn}
            onChange={(e) => setLabelColumn(e.target.value)}
            placeholder="labels"
            disabled={loading}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <Input
            label="Method 1 Name"
            value={method1Name}
            onChange={(e) => setMethod1Name(e.target.value)}
            placeholder="Method 1"
            disabled={loading}
          />
          <Input
            label="Method 2 Name"
            value={method2Name}
            onChange={(e) => setMethod2Name(e.target.value)}
            placeholder="Method 2"
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

        <Button onClick={handleCompare} loading={loading} disabled={!csvFile1 || !csvFile2 || loading}>
          Evaluate with LLM Judge
        </Button>

        {result && (
          <div className="mt-6 space-y-6">
            {/* Summary */}
            <div className="p-4 bg-blue-50 rounded-lg space-y-1">
              <p className="text-sm text-blue-800">
                Evaluated <strong>{result.total_questions_evaluated}</strong> questions.
              </p>
              {result.ignored_data &&
                (result.ignored_data.ids_only_in_file1 > 0 ||
                  result.ignored_data.ids_only_in_file2 > 0 ||
                  result.ignored_data.skipped_no_labels > 0 ||
                  result.ignored_data.skipped_no_text > 0) && (
                  <p className="text-sm text-blue-700">
                    Ignored: <strong>{result.ignored_data.ids_only_in_file1}</strong> IDs only in file 1,{' '}
                    <strong>{result.ignored_data.ids_only_in_file2}</strong> only in file 2,{' '}
                    <strong>{result.ignored_data.skipped_no_labels}</strong> missing labels,{' '}
                    <strong>{result.ignored_data.skipped_no_text}</strong> missing text.
                  </p>
                )}
            </div>
            {result.pairwise_comparison && (
              <div className="p-4 bg-emerald-50 rounded-lg space-y-1">
                <p className="text-sm text-emerald-800">
                  Pairwise comparison (avg of 4 metrics):{' '}
                  <strong>{result.method1_name}</strong> wins{' '}
                  <strong>{result.pairwise_comparison.method1_wins}</strong>,{' '}
                  <strong>{result.method2_name}</strong> wins{' '}
                  <strong>{result.pairwise_comparison.method2_wins}</strong>,{' '}
                  ties <strong>{result.pairwise_comparison.ties}</strong>.
                </p>
              </div>
            )}
            {result.pairwise_judge && (
              <div className="p-4 bg-purple-50 rounded-lg space-y-1">
                <p className="text-sm text-purple-800">
                  Pairwise LLM judge:{' '}
                  <strong>{result.pairwise_judge.method1_name}</strong> wins{' '}
                  <strong>{result.pairwise_judge.method1_wins}</strong>,{' '}
                  <strong>{result.pairwise_judge.method2_name}</strong> wins{' '}
                  <strong>{result.pairwise_judge.method2_wins}</strong>,{' '}
                  ties <strong>{result.pairwise_judge.ties}</strong>.
                </p>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-4">
              <div className="flex-1 min-w-0">
                <RunStatistics stats={result} className="mb-0" />
              </div>
              <Button
                onClick={handleDownloadResults}
                disabled={!result?.question_metrics?.length}
                variant="secondary"
              >
                Download evaluation results (CSV)
              </Button>
            </div>

            {/* Average Metrics */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Average Metrics Across All Questions</h3>
              <div className="grid grid-cols-2 gap-6">
                {/* Method 1 Averages */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold mb-3">{result.method1_name}</h4>
                  <div className="space-y-2">
                    <MetricRow label="Correctness" value={result.average_metrics.method1.correctness} max={5} />
                    <MetricRow label="Completeness" value={result.average_metrics.method1.completeness} max={5} />
                    <MetricRow label="Generalization" value={result.average_metrics.method1.generalization} max={5} />
                    <MetricRow label="Consistency" value={result.average_metrics.method1.consistency} max={5} />
                  </div>
                </div>

                {/* Method 2 Averages */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold mb-3">{result.method2_name}</h4>
                  <div className="space-y-2">
                    <MetricRow label="Correctness" value={result.average_metrics.method2.correctness} max={5} />
                    <MetricRow label="Completeness" value={result.average_metrics.method2.completeness} max={5} />
                    <MetricRow label="Generalization" value={result.average_metrics.method2.generalization} max={5} />
                    <MetricRow label="Consistency" value={result.average_metrics.method2.consistency} max={5} />
                  </div>
                </div>
              </div>
            </div>

            {/* Question Details Table */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Per-Question Evaluation Details</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        Question ID
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        {result.method1_name}
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        {result.method2_name}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {result.question_metrics.slice(0, 10).map((item: any, idx: number) => (
                      <tr key={idx} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-sm font-medium text-gray-900">{item.id}</td>
                        <td className="px-4 py-3 text-sm">
                          <div className="space-y-1">
                            <div className="text-xs text-gray-500">
                              Labels: {item.method1.labels.join(', ') || 'None'}
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                              <div>Cor: {item.method1.correctness}/5</div>
                              <div>Comp: {item.method1.completeness}/5</div>
                              <div>Gen: {item.method1.generalization}/5</div>
                              <div>Cons: {item.method1.consistency}/5</div>
                            </div>
                            <div className="text-xs text-gray-600 italic mt-1">
                              {item.method1.reasoning}
                            </div>
                          </div>
                        </td>
                        <td className="px-4 py-3 text-sm">
                          <div className="space-y-1">
                            <div className="text-xs text-gray-500">
                              Labels: {item.method2.labels.join(', ') || 'None'}
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                              <div>Cor: {item.method2.correctness}/5</div>
                              <div>Comp: {item.method2.completeness}/5</div>
                              <div>Gen: {item.method2.generalization}/5</div>
                              <div>Cons: {item.method2.consistency}/5</div>
                            </div>
                            <div className="text-xs text-gray-600 italic mt-1">
                              {item.method2.reasoning}
                            </div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.question_metrics.length > 10 && (
                  <p className="mt-2 text-sm text-gray-500 text-center">
                    Showing first 10 of {result.question_metrics.length} questions
                  </p>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: number;
  max: number;
}

function MetricRow({ label, value, max }: MetricRowProps) {
  const percentage = (value / max) * 100;
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-700">{label}</span>
        <span className="font-medium">{value.toFixed(2)}/{max}</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-600 h-2 rounded-full"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
