import { useState } from 'react';
import FileUpload from '../components/FileUpload';
import Button from '../components/Button';
import Input from '../components/Input';
import Select from '../components/Select';
import RunStatistics from '../components/RunStatistics';
import { downloadFile, evaluationApi } from '../services/api';

const METRIC_MAX = 5;
const METRICS = [
  {
    key: 'intent_alignment_score',
    label: 'Intent Alignment Score (IAS)',
    shortLabel: 'IAS',
    description: 'Matches the primary intent without drifting to secondary topics.',
  },
  {
    key: 'concept_completeness_score',
    label: 'Concept Completeness Score (CCS)',
    shortLabel: 'CCS',
    description: 'Covers all essential concepts without rewarding extra noise.',
  },
  {
    key: 'noise_redundancy_penalty',
    label: 'Noise & Redundancy Penalty (NRP)',
    shortLabel: 'NRP',
    description: 'Higher is cleaner: fewer redundant, vague, or generic labels.',
  },
  {
    key: 'terminology_normalization_score',
    label: 'Terminology Normalization Score (TNS)',
    shortLabel: 'TNS',
    description: 'Uses canonical terms with consistent, taxonomy-like naming.',
  },
  {
    key: 'audit_usefulness_score',
    label: 'Audit Usefulness Score (AUS)',
    shortLabel: 'AUS',
    description: 'Actionable for audit scoping, testing, or evidence mapping.',
  },
  {
    key: 'control_mapping_clarity_score',
    label: 'Control-Mapping Clarity Score (CMCS)',
    shortLabel: 'CMCS',
    description: 'Clearly indicates the control domain (IAM, logging, incident, etc.).',
  },
] as const;

const PAPER_DIMENSIONS = [
  { key: 'correctness', label: 'Correctness' },
  { key: 'completeness', label: 'Completeness' },
  { key: 'clarity', label: 'Clarity' },
  { key: 'faithfulness', label: 'Faithfulness' },
  { key: 'overall', label: 'Overall' },
] as const;

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
  const [randomSeed, setRandomSeed] = useState('1234');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [paperResult, setPaperResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const buildFormData = () => {
    if (!csvFile1 || !csvFile2) {
      setError('Please upload both CSV files');
      return null;
    }

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
    return formData;
  };

  const runStandardEvaluation = async () => {
    const formData = buildFormData();
    if (!formData) return;
    setLoading(true);
    setError(null);
    try {
      const response = await evaluationApi.compare(formData);
      setResult(response.evaluation_results);
      setPaperResult(null);
    } catch (err: any) {
      setError(err.message || 'Failed to evaluate files');
    } finally {
      setLoading(false);
    }
  };

  const runPaperEvaluation = async (mode: 'pairwise' | 'absolute' | 'both') => {
    const formData = buildFormData();
    if (!formData) return;
    const seed = Number.parseInt(randomSeed, 10);
    if (!Number.isNaN(seed)) {
      formData.append('random_seed', String(seed));
    }
    const runPairwise = mode === 'pairwise' || mode === 'both';
    const runAbsolute = mode === 'absolute' || mode === 'both';
    formData.append('run_pairwise', String(runPairwise));
    formData.append('run_absolute', String(runAbsolute));
    setLoading(true);
    setError(null);
    try {
      const response = await evaluationApi.paper(formData);
      setPaperResult(response.evaluation_results);
      setResult(null);
    } catch (err: any) {
      setError(err.message || 'Failed to evaluate files');
    } finally {
      setLoading(false);
    }
  };

  const runBothEvaluations = async () => {
    const formDataStandard = buildFormData();
    const formDataPaper = buildFormData();
    if (!formDataStandard || !formDataPaper) return;
    const seed = Number.parseInt(randomSeed, 10);
    if (!Number.isNaN(seed)) {
      formDataPaper.append('random_seed', String(seed));
    }
    formDataPaper.append('run_pairwise', 'true');
    formDataPaper.append('run_absolute', 'true');
    setLoading(true);
    setError(null);
    try {
      const standardResponse = await evaluationApi.compare(formDataStandard);
      setResult(standardResponse.evaluation_results);
      const paperResponse = await evaluationApi.paper(formDataPaper);
      setPaperResult(paperResponse.evaluation_results);
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
    const headers = [
      'Question ID',
      ...METRICS.map((metric) => `${m1} ${metric.label}`),
      `${m1} Reasoning`,
      `${m1} Labels`,
      ...METRICS.map((metric) => `${m2} ${metric.label}`),
      `${m2} Reasoning`,
      `${m2} Labels`,
    ];
    const rows: string[] = [headers.map(escapeCsv).join(',')];
    for (const q of result.question_metrics) {
      const r1 = q.method1;
      const r2 = q.method2;
      rows.push([
        escapeCsv(q.id),
        ...METRICS.map((metric) => r1?.[metric.key] ?? ''),
        escapeCsv(r1.reasoning ?? ''),
        escapeCsv((r1.labels ?? []).join(', ')),
        ...METRICS.map((metric) => r2?.[metric.key] ?? ''),
        escapeCsv(r2.reasoning ?? ''),
        escapeCsv((r2.labels ?? []).join(', ')),
      ].join(','));
    }
    const avg = result.average_metrics;
    if (avg?.method1 && avg?.method2) {
      rows.push('');
      rows.push([
        'AVERAGE',
        ...METRICS.map((metric) =>
          avg.method1[metric.key] != null ? avg.method1[metric.key].toFixed(2) : ''
        ),
        '',
        '',
        ...METRICS.map((metric) =>
          avg.method2[metric.key] != null ? avg.method2[metric.key].toFixed(2) : ''
        ),
        '',
        '',
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

  const handleDownloadPaperFile = async (filePath: string) => {
    if (!filePath) return;
    try {
      await downloadFile(filePath);
    } catch (err: any) {
      setError(err.message || 'Failed to download file');
    }
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
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-800">Run options</h2>
            <p className="text-xs text-slate-600">
              Run standard IAS/CCS/NRP/TNS/AUS/CMCS, paper-ready pairwise + absolute scoring, or both.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Input
              label="Random Seed (paper-ready)"
              type="number"
              value={randomSeed}
              onChange={(e) => setRandomSeed(e.target.value)}
              placeholder="1234"
              disabled={loading}
            />
            <div className="md:col-span-2 text-xs text-slate-600 flex items-end">
              The seed controls randomized A/B ordering to reduce positional bias.
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button onClick={runStandardEvaluation} loading={loading} disabled={!csvFile1 || !csvFile2 || loading}>
              Run standard evaluation
            </Button>
            <Button
              onClick={() => runPaperEvaluation('pairwise')}
              loading={loading}
              disabled={!csvFile1 || !csvFile2 || loading}
              variant="secondary"
            >
              Run paper pairwise
            </Button>
            <Button
              onClick={() => runPaperEvaluation('absolute')}
              loading={loading}
              disabled={!csvFile1 || !csvFile2 || loading}
              variant="secondary"
            >
              Run paper absolute
            </Button>
            <Button
              onClick={() => runPaperEvaluation('both')}
              loading={loading}
              disabled={!csvFile1 || !csvFile2 || loading}
              variant="secondary"
            >
              Run paper (both)
            </Button>
            <Button
              onClick={runBothEvaluations}
              loading={loading}
              disabled={!csvFile1 || !csvFile2 || loading}
              variant="secondary"
            >
              Run all (standard + paper)
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-semibold text-slate-800">Standard metrics guide</h2>
            <span className="text-xs text-slate-500">Scale: 0-{METRIC_MAX}</span>
          </div>
          <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
            {METRICS.map((metric) => (
              <div key={metric.key} className="rounded-md border border-slate-200 bg-white p-3">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-semibold text-slate-900">{metric.label}</p>
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-600 bg-slate-100 rounded px-2 py-0.5">
                    {metric.shortLabel}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-600">{metric.description}</p>
              </div>
            ))}
          </div>
        </div>
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
                    {METRICS.map((metric) => (
                      <MetricRow
                        key={`m1-${metric.key}`}
                        label={metric.label}
                        value={result.average_metrics.method1[metric.key]}
                        max={METRIC_MAX}
                      />
                    ))}
                  </div>
                </div>

                {/* Method 2 Averages */}
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold mb-3">{result.method2_name}</h4>
                  <div className="space-y-2">
                    {METRICS.map((metric) => (
                      <MetricRow
                        key={`m2-${metric.key}`}
                        label={metric.label}
                        value={result.average_metrics.method2[metric.key]}
                        max={METRIC_MAX}
                      />
                    ))}
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
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              {METRICS.map((metric) => (
                                <div key={`m1-${metric.key}`}>
                                  {metric.shortLabel}: {item.method1[metric.key]}/{METRIC_MAX}
                                </div>
                              ))}
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
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              {METRICS.map((metric) => (
                                <div key={`m2-${metric.key}`}>
                                  {metric.shortLabel}: {item.method2[metric.key]}/{METRIC_MAX}
                                </div>
                              ))}
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

        {paperResult && (
          <div className="mt-6 space-y-6">
            <div className="p-4 bg-indigo-50 rounded-lg space-y-1">
              <p className="text-sm text-indigo-900">
                Evaluated <strong>{paperResult.total_questions_evaluated}</strong> questions.
              </p>
              {paperResult.ignored_data &&
                (paperResult.ignored_data.ids_only_in_file1 > 0 ||
                  paperResult.ignored_data.ids_only_in_file2 > 0 ||
                  paperResult.ignored_data.skipped_no_labels > 0 ||
                  paperResult.ignored_data.skipped_no_text > 0) && (
                  <p className="text-sm text-indigo-800">
                    Ignored: <strong>{paperResult.ignored_data.ids_only_in_file1}</strong> IDs only in file 1,{' '}
                    <strong>{paperResult.ignored_data.ids_only_in_file2}</strong> only in file 2,{' '}
                    <strong>{paperResult.ignored_data.skipped_no_labels}</strong> missing labels,{' '}
                    <strong>{paperResult.ignored_data.skipped_no_text}</strong> missing text.
                  </p>
                )}
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <div className="flex-1 min-w-0">
                <RunStatistics stats={paperResult} className="mb-0" />
              </div>
              {paperResult.files && (
                <div className="flex flex-wrap gap-2">
                  {paperResult.files.pairwise_judgments_csv && (
                    <Button
                      variant="secondary"
                      onClick={() => handleDownloadPaperFile(paperResult.files.pairwise_judgments_csv)}
                    >
                      Pairwise judgments CSV
                    </Button>
                  )}
                  {paperResult.files.absolute_scores_csv && (
                    <Button
                      variant="secondary"
                      onClick={() => handleDownloadPaperFile(paperResult.files.absolute_scores_csv)}
                    >
                      Absolute scores CSV
                    </Button>
                  )}
                  {paperResult.files.pairwise_summary_csv && (
                    <Button
                      variant="secondary"
                      onClick={() => handleDownloadPaperFile(paperResult.files.pairwise_summary_csv)}
                    >
                      Pairwise summary CSV
                    </Button>
                  )}
                  {paperResult.files.average_scores_csv && (
                    <Button
                      variant="secondary"
                      onClick={() => handleDownloadPaperFile(paperResult.files.average_scores_csv)}
                    >
                      Average scores CSV
                    </Button>
                  )}
                  {paperResult.files.dimension_breakdown_csv && (
                    <Button
                      variant="secondary"
                      onClick={() => handleDownloadPaperFile(paperResult.files.dimension_breakdown_csv)}
                    >
                      Dimension breakdown CSV
                    </Button>
                  )}
                  {paperResult.files.json && (
                    <Button
                      variant="secondary"
                      onClick={() => handleDownloadPaperFile(paperResult.files.json)}
                    >
                      Reproducibility JSON
                    </Button>
                  )}
                </div>
              )}
            </div>

            {paperResult.pairwise_summary && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-white rounded-lg border border-gray-200">
                  <p className="text-xs uppercase tracking-wide text-gray-500">
                    Win rate ({paperResult.method1_name})
                  </p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {paperResult.pairwise_summary.win_rate_pct.toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-600">
                    {paperResult.pairwise_summary.wins} wins
                  </p>
                </div>
                <div className="p-4 bg-white rounded-lg border border-gray-200">
                  <p className="text-xs uppercase tracking-wide text-gray-500">
                    Loss rate ({paperResult.method2_name})
                  </p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {paperResult.pairwise_summary.loss_rate_pct.toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-600">
                    {paperResult.pairwise_summary.losses} losses
                  </p>
                </div>
                <div className="p-4 bg-white rounded-lg border border-gray-200">
                  <p className="text-xs uppercase tracking-wide text-gray-500">Tie rate</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {paperResult.pairwise_summary.tie_rate_pct.toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-600">
                    {paperResult.pairwise_summary.ties} ties
                  </p>
                </div>
              </div>
            )}

            {paperResult.average_scores && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold mb-3">{paperResult.method1_name} (Mean +/- Std)</h4>
                  <div className="space-y-2 text-sm">
                    {PAPER_DIMENSIONS.map((dim) => (
                      <div key={`m1-${dim.key}`} className="flex justify-between">
                        <span className="text-gray-700">{dim.label}</span>
                        <span className="font-medium">
                          {paperResult.average_scores.method1[dim.key].mean.toFixed(2)} +/-{' '}
                          {paperResult.average_scores.method1[dim.key].std.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold mb-3">{paperResult.method2_name} (Mean +/- Std)</h4>
                  <div className="space-y-2 text-sm">
                    {PAPER_DIMENSIONS.map((dim) => (
                      <div key={`m2-${dim.key}`} className="flex justify-between">
                        <span className="text-gray-700">{dim.label}</span>
                        <span className="font-medium">
                          {paperResult.average_scores.method2[dim.key].mean.toFixed(2)} +/-{' '}
                          {paperResult.average_scores.method2[dim.key].std.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {paperResult.dimension_breakdown && (
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <h4 className="font-semibold mb-3">Dimension-wise comparison</h4>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 text-sm">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Dimension</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                          {paperResult.method1_name}
                        </th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                          {paperResult.method2_name}
                        </th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Delta</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {paperResult.dimension_breakdown.map((row: any, idx: number) => (
                        <tr key={idx}>
                          <td className="px-4 py-2 font-medium text-gray-800">{row.dimension}</td>
                          <td className="px-4 py-2">{row.method1_mean.toFixed(2)}</td>
                          <td className="px-4 py-2">{row.method2_mean.toFixed(2)}</td>
                          <td className="px-4 py-2">{row.delta.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {paperResult.statistical_test && (
              <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
                <h4 className="font-semibold mb-2">Statistical test (binomial)</h4>
                <div className="text-sm text-slate-700 space-y-1">
                  <div>
                    p-value:{' '}
                    {paperResult.statistical_test.p_value != null
                      ? paperResult.statistical_test.p_value.toExponential(2)
                      : 'N/A'}
                  </div>
                  <div>Alpha: {paperResult.statistical_test.alpha}</div>
                  <div>
                    Result:{' '}
                    <span className={paperResult.statistical_test.significant ? 'text-green-700 font-semibold' : ''}>
                      {paperResult.statistical_test.interpretation}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {paperResult.latex_table && (
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <h4 className="font-semibold mb-3">LaTeX table (dimension breakdown)</h4>
                <pre className="text-xs bg-gray-50 p-3 rounded border border-gray-200 overflow-x-auto">
                  {paperResult.latex_table}
                </pre>
              </div>
            )}
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
