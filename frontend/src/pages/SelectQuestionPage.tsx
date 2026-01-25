import { useEffect, useMemo, useRef, useState } from 'react';
import FileUpload from '../components/FileUpload';
import Button from '../components/Button';
import Input from '../components/Input';
import Select from '../components/Select';
import RunStatistics from '../components/RunStatistics';
import { selectQuestionApi, SelectQuestionResponse } from '../services/api';

type Method = 'bm25' | 'embedding' | 'label_embedding';
type SortKey = 'original' | 'score' | 'id' | 'question' | 'labels' | 'matched_labels';
type SortDir = 'asc' | 'desc';

export default function SelectQuestionPage() {
  const [method, setMethod] = useState<Method>('bm25');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [userNeed, setUserNeed] = useState('');
  const [textColumn, setTextColumn] = useState('text');
  const [idColumn, setIdColumn] = useState('id');
  const [labelColumn, setLabelColumn] = useState('labels');
  const [topK, setTopK] = useState(5);

  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-3-large');
  const [embedType, setEmbedType] = useState<'open_ai' | 'ollama' | 'huggingface'>('open_ai');
  const [batchSize, setBatchSize] = useState(32);
  const [apiKey, setApiKey] = useState('');
  const [endpoint, setEndpoint] = useState('');

  const [topKLabels, setTopKLabels] = useState(5);
  const [topKQuestions, setTopKQuestions] = useState(5);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SelectQuestionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedRows, setSelectedRows] = useState<Set<string>>(new Set());
  const [rowIds, setRowIds] = useState<string[]>([]);
  const [sortKey, setSortKey] = useState<SortKey>('original');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const selectAllRef = useRef<HTMLInputElement | null>(null);

  const createRowId = (id: string, idx: number) => {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID();
    }
    return `${id}-${idx}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  };

  const resultRows = useMemo(() => {
    if (!result) return [];
    return result.results.map((row, idx) => ({
      row,
      rowId: rowIds[idx] ?? `${row.id}-${idx}`,
      originalIndex: idx,
    }));
  }, [result, rowIds]);

  const sortedRows = useMemo(() => {
    if (sortKey === 'original') return resultRows;
    const direction = sortDir === 'asc' ? 1 : -1;
    const rows = [...resultRows];
    rows.sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'score') {
        const aScore = a.row.score;
        const bScore = b.row.score;
        if (aScore == null && bScore == null) cmp = 0;
        else if (aScore == null) cmp = 1;
        else if (bScore == null) cmp = -1;
        else cmp = aScore - bScore;
      } else {
        const getString = (val: string | undefined | null) => (val ?? '').toString().toLowerCase();
        const aVal =
          sortKey === 'id'
            ? getString(a.row.id)
            : sortKey === 'question'
              ? getString(a.row.question)
              : sortKey === 'labels'
                ? getString((a.row.labels ?? []).join(', '))
                : getString((a.row.matched_labels ?? []).join(', '));
        const bVal =
          sortKey === 'id'
            ? getString(b.row.id)
            : sortKey === 'question'
              ? getString(b.row.question)
              : sortKey === 'labels'
                ? getString((b.row.labels ?? []).join(', '))
                : getString((b.row.matched_labels ?? []).join(', '));
        cmp = aVal.localeCompare(bVal, undefined, { numeric: true, sensitivity: 'base' });
      }
      if (cmp === 0) return a.originalIndex - b.originalIndex;
      return cmp * direction;
    });
    return rows;
  }, [resultRows, sortDir, sortKey]);

  const selectedCount = resultRows.reduce(
    (count, item) => (selectedRows.has(item.rowId) ? count + 1 : count),
    0
  );
  const allSelected = resultRows.length > 0 && selectedCount === resultRows.length;

  useEffect(() => {
    if (!selectAllRef.current) return;
    selectAllRef.current.indeterminate = selectedCount > 0 && !allSelected;
  }, [selectedCount, allSelected]);

  const escapeCsv = (v: string | number | null | undefined): string => {
    if (v == null) return '';
    const s = String(v);
    if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };

  const handleToggleRow = (key: string) => {
    setSelectedRows((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const handleToggleAll = () => {
    if (!resultRows.length) return;
    setSelectedRows((prev) => {
      const keys = resultRows.map((item) => item.rowId);
      const isAllSelected = keys.length > 0 && keys.every((key) => prev.has(key));
      if (isAllSelected) {
        return new Set();
      }
      return new Set(keys);
    });
  };

  const handleDownloadSelected = () => {
    if (!result) return;
    const selected = sortedRows.filter((item) => selectedRows.has(item.rowId)).map((item) => item.row);
    if (!selected.length) return;
    const headers = ['id', 'question', 'score', 'labels', 'matched_labels'];
    const rows: string[] = [headers.map(escapeCsv).join(',')];
    selected.forEach((row) => {
      rows.push([
        escapeCsv(row.id),
        escapeCsv(row.question),
        escapeCsv(row.score != null ? row.score.toFixed(4) : ''),
        escapeCsv((row.labels ?? []).join(', ')),
        escapeCsv((row.matched_labels ?? []).join(', ')),
      ].join(','));
    });
    const csv = '\uFEFF' + rows.join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `selected-questions-${result.method}-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleRun = async () => {
    if (!csvFile) {
      setError('Please upload a CSV file');
      return;
    }
    if (!userNeed.trim()) {
      setError('Please enter the user requirement');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedRows(new Set());
    setRowIds([]);
    setSortKey('original');
    setSortDir('desc');

    try {
      const formData = new FormData();
      formData.append('file', csvFile);
      formData.append('user_need', userNeed.trim());
      formData.append('text_column', textColumn.trim() || 'text');
      formData.append('id_column', idColumn.trim() || 'id');

      if (method === 'bm25') {
        if (labelColumn.trim()) formData.append('label_column', labelColumn.trim());
        formData.append('top_k', topK.toString());
        const response = await selectQuestionApi.bm25(formData);
        setResult(response);
        setRowIds(response.results.map((row, idx) => createRowId(row.id, idx)));
        setSelectedRows(new Set());
        setSortKey('original');
        setSortDir('desc');
      }

      if (method === 'embedding') {
        if (labelColumn.trim()) formData.append('label_column', labelColumn.trim());
        formData.append('embedding_model', embeddingModel);
        formData.append('embed_type', embedType);
        formData.append('batch_size', batchSize.toString());
        formData.append('top_k', topK.toString());
        if (apiKey.trim()) formData.append('api_key', apiKey.trim());
        if (endpoint.trim()) formData.append('endpoint', endpoint.trim());
        const response = await selectQuestionApi.embedding(formData);
        setResult(response);
        setRowIds(response.results.map((row, idx) => createRowId(row.id, idx)));
        setSelectedRows(new Set());
        setSortKey('original');
        setSortDir('desc');
      }

      if (method === 'label_embedding') {
        formData.append('label_column', labelColumn.trim() || 'labels');
        formData.append('embedding_model', embeddingModel);
        formData.append('embed_type', embedType);
        formData.append('batch_size', batchSize.toString());
        formData.append('top_k_labels', topKLabels.toString());
        formData.append('top_k_questions', topKQuestions.toString());
        if (apiKey.trim()) formData.append('api_key', apiKey.trim());
        if (endpoint.trim()) formData.append('endpoint', endpoint.trim());
        const response = await selectQuestionApi.labelEmbedding(formData);
        setResult(response);
        setRowIds(response.results.map((row, idx) => createRowId(row.id, idx)));
        setSelectedRows(new Set());
        setSortKey('original');
        setSortDir('desc');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to select questions');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Select Questions</h1>
        <p className="mt-2 text-gray-600">
          Retrieve the best-matching questions using BM25 or embedding similarity.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div className="grid grid-cols-2 gap-4">
          <Select
            label="Method"
            value={method}
            onChange={(e) => setMethod(e.target.value as Method)}
            options={[
              { value: 'bm25', label: 'BM25 (Lexical)' },
              { value: 'embedding', label: 'Embedding Similarity' },
              { value: 'label_embedding', label: 'Label Embedding + Match' },
            ]}
          />
          <Input
            label="Top K (Questions)"
            type="number"
            value={method === 'label_embedding' ? topKQuestions : topK}
            onChange={(e) => {
              const val = parseInt(e.target.value) || 1;
              if (method === 'label_embedding') {
                setTopKQuestions(val);
              } else {
                setTopK(val);
              }
            }}
          />
        </div>

        <FileUpload
          accept=".csv"
          file={csvFile}
          onChange={setCsvFile}
          label="Upload CSV File"
          disabled={loading}
        />

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">User Requirement</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
            value={userNeed}
            onChange={(e) => setUserNeed(e.target.value)}
            placeholder="Describe the user need or requirement..."
            disabled={loading}
          />
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Input
            label="Text Column"
            value={textColumn}
            onChange={(e) => setTextColumn(e.target.value)}
            disabled={loading}
          />
          <Input
            label="ID Column"
            value={idColumn}
            onChange={(e) => setIdColumn(e.target.value)}
            disabled={loading}
          />
          <Input
            label={method === 'label_embedding' ? 'Label Column (required)' : 'Label Column (optional)'}
            value={labelColumn}
            onChange={(e) => setLabelColumn(e.target.value)}
            placeholder={method === 'label_embedding' ? 'labels' : 'optional'}
            disabled={loading}
          />
        </div>

        {method !== 'bm25' && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <Input
                label="Embedding Model"
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                disabled={loading}
              />
              <Select
                label="Embedding Type"
                value={embedType}
                onChange={(e) => setEmbedType(e.target.value as any)}
                options={[
                  { value: 'open_ai', label: 'OpenAI' },
                  { value: 'ollama', label: 'Ollama' },
                  { value: 'huggingface', label: 'HuggingFace' },
                ]}
              />
              <Input
                label="Batch Size"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                disabled={loading}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Input
                label="API Key (optional)"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Leave empty to use environment variable"
                disabled={loading}
              />
              <Input
                label="Endpoint (optional, e.g. Ollama base URL)"
                value={endpoint}
                onChange={(e) => setEndpoint(e.target.value)}
                placeholder="http://localhost:11434"
                disabled={loading}
              />
            </div>
          </div>
        )}

        {method === 'label_embedding' && (
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Top K Labels"
              type="number"
              value={topKLabels}
              onChange={(e) => setTopKLabels(parseInt(e.target.value) || 1)}
              disabled={loading}
            />
            <div className="text-sm text-gray-500 flex items-end">
              Uses top labels to filter matching questions.
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <Button onClick={handleRun} loading={loading} disabled={loading || !csvFile || !userNeed.trim()}>
          Run Selection
        </Button>
      </div>

      {result && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Results</h2>
              <p className="text-sm text-gray-600">
                {result.total_results} question{result.total_results === 1 ? '' : 's'} returned
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-sm text-gray-600">
                Selected: <span className="font-medium text-gray-900">{selectedCount}</span>
              </div>
              <Button
                onClick={handleDownloadSelected}
                disabled={selectedCount === 0}
                variant="secondary"
              >
                Download selected (CSV)
              </Button>
            </div>
          </div>

          <RunStatistics stats={result} />

          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Select
              label="Order By"
              value={sortKey}
              onChange={(e) => setSortKey(e.target.value as SortKey)}
              options={[
                { value: 'original', label: 'Original order' },
                { value: 'score', label: 'Score' },
                { value: 'id', label: 'ID' },
                { value: 'question', label: 'Question' },
                { value: 'labels', label: 'Labels' },
                { value: 'matched_labels', label: 'Matched Labels' },
              ]}
            />
            <Select
              label="Direction"
              value={sortDir}
              onChange={(e) => setSortDir(e.target.value as SortDir)}
              options={[
                { value: 'desc', label: 'Descending' },
                { value: 'asc', label: 'Ascending' },
              ]}
              disabled={sortKey === 'original'}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">
                    <input
                      ref={selectAllRef}
                      type="checkbox"
                      className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      checked={allSelected}
                      onChange={handleToggleAll}
                      aria-label="Select all questions"
                    />
                  </th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">ID</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Score</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Question</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-600">Labels</th>
                  {result.method === 'label_embedding' && (
                    <th className="px-4 py-2 text-left font-medium text-gray-600">Matched Labels</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {sortedRows.map((item) => (
                  <tr key={item.rowId}>
                    <td className="px-4 py-2 text-gray-700">
                      <input
                        type="checkbox"
                        className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        checked={selectedRows.has(item.rowId)}
                        onChange={() => handleToggleRow(item.rowId)}
                        aria-label={`Select question ${item.row.id}`}
                      />
                    </td>
                    <td className="px-4 py-2 text-gray-700">{item.row.id}</td>
                    <td className="px-4 py-2 text-gray-700">
                      {item.row.score !== undefined && item.row.score !== null ? item.row.score.toFixed(4) : '-'}
                    </td>
                    <td className="px-4 py-2 text-gray-700">{item.row.question}</td>
                    <td className="px-4 py-2 text-gray-700">
                      {item.row.labels && item.row.labels.length ? item.row.labels.join(', ') : '-'}
                    </td>
                    {result.method === 'label_embedding' && (
                      <td className="px-4 py-2 text-gray-700">
                        {item.row.matched_labels && item.row.matched_labels.length ? item.row.matched_labels.join(', ') : '-'}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
