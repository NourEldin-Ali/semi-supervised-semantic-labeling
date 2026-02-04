import { useState } from 'react';
import Button from '../components/Button';
import FileUpload from '../components/FileUpload';
import Input from '../components/Input';
import Select from '../components/Select';
import RunStatistics from '../components/RunStatistics';
import { QuestionSetCompareResponse, QuestionSetScoreResponse, selectQuestionApi } from '../services/api';

export default function QuestionJudgePage() {
  const [llmModel, setLlmModel] = useState('gpt-5.2-2025-12-11');
  const [llmType, setLlmType] = useState('open_ai');
  const [apiKey, setApiKey] = useState('');

  const [setCompareNeedText, setSetCompareNeedText] = useState('');
  const [setCompareFileA, setSetCompareFileA] = useState<File | null>(null);
  const [setCompareFileB, setSetCompareFileB] = useState<File | null>(null);
  const [setCompareTextColumn, setSetCompareTextColumn] = useState('question');
  const [setCompareBusy, setSetCompareBusy] = useState(false);
  const [setCompareErrorMsg, setSetCompareErrorMsg] = useState<string | null>(null);
  const [setCompareData, setSetCompareData] = useState<QuestionSetCompareResponse | null>(null);

  const [setFileNeedText, setSetFileNeedText] = useState('');
  const [setFileCsv, setSetFileCsv] = useState<File | null>(null);
  const [setFileTextColumn, setSetFileTextColumn] = useState('question');
  const [setFileScoreBusy, setSetFileScoreBusy] = useState(false);
  const [setFileScoreErrorMsg, setSetFileScoreErrorMsg] = useState<string | null>(null);
  const [setFileScoreData, setSetFileScoreData] = useState<QuestionSetScoreResponse | null>(null);

  const handleSetScoreFile = async () => {
    if (!setFileNeedText.trim() || !setFileCsv) {
      setSetFileScoreErrorMsg('Please provide the user need and upload a CSV file.');
      return;
    }
    setSetFileScoreBusy(true);
    setSetFileScoreErrorMsg(null);
    setSetFileScoreData(null);

    try {
      const formData = new FormData();
      formData.append('user_need', setFileNeedText.trim());
      formData.append('file', setFileCsv);
      formData.append('text_column', setFileTextColumn.trim() || 'question');
      formData.append('llm_model', llmModel);
      formData.append('llm_type', llmType);
      if (apiKey.trim()) formData.append('api_key', apiKey.trim());
      const response = await selectQuestionApi.scoreSetFile(formData);
      setSetFileScoreData(response);
    } catch (err: any) {
      setSetFileScoreErrorMsg(err.message || 'Failed to score question set (file)');
    } finally {
      setSetFileScoreBusy(false);
    }
  };

  const handleSetCompare = async () => {
    if (!setCompareNeedText.trim() || !setCompareFileA || !setCompareFileB) {
      setSetCompareErrorMsg('Please provide the user need and upload both CSV files.');
      return;
    }
    setSetCompareBusy(true);
    setSetCompareErrorMsg(null);
    setSetCompareData(null);

    try {
      const formData = new FormData();
      formData.append('user_need', setCompareNeedText.trim());
      formData.append('file_a', setCompareFileA);
      formData.append('file_b', setCompareFileB);
      formData.append('text_column', setCompareTextColumn.trim() || 'question');
      formData.append('llm_model', llmModel);
      formData.append('llm_type', llmType);
      if (apiKey.trim()) formData.append('api_key', apiKey.trim());
      const response = await selectQuestionApi.compareSetsFile(formData);
      setSetCompareData(response);
    } catch (err: any) {
      setSetCompareErrorMsg(err.message || 'Failed to compare question sets');
    } finally {
      setSetCompareBusy(false);
    }
  };

  const renderSetWinner = (winner: 'A' | 'B' | 'tie') => {
    if (winner === 'A') return 'Set A';
    if (winner === 'B') return 'Set B';
    return 'Tie';
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Question Judge</h1>
        <p className="mt-2 text-gray-600">
          Score a generated question against a user need, or compare two questions.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <h2 className="text-xl font-semibold text-gray-900">LLM Settings</h2>
        <div className="grid grid-cols-2 gap-4">
          <Input
            label="LLM Model"
            value={llmModel}
            onChange={(e) => setLlmModel(e.target.value)}
            placeholder="gpt-5.2-2025-12-11"
          />
          <Select
            label="LLM Type"
            value={llmType}
            onChange={(e) => setLlmType(e.target.value)}
            options={[
              { value: 'open_ai', label: 'OpenAI' },
              { value: 'groq_ai', label: 'Groq AI' },
              { value: 'ollama', label: 'Ollama' },
              { value: 'anthropic', label: 'Anthropic' },
            ]}
          />
        </div>
        <Input
          label="API Key (optional, can use .env)"
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Leave empty to use .env"
        />
      </div>

      {/* <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <h2 className="text-xl font-semibold text-gray-900">Score a Single Question</h2>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">User Need</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
            value={scoreNeed}
            onChange={(e) => setScoreNeed(e.target.value)}
            placeholder="Describe the user requirement..."
            disabled={scoreLoading}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Generated Question</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
            value={scoreQuestion}
            onChange={(e) => setScoreQuestion(e.target.value)}
            placeholder="Paste the generated question..."
            disabled={scoreLoading}
          />
        </div>
        {scoreError && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {scoreError}
          </div>
        )}
        <Button onClick={handleScore} loading={scoreLoading} disabled={scoreLoading}>
          Score Question
        </Button>

        {scoreResult && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-700">Score</p>
              <div className="text-4xl font-bold text-blue-900">
                {scoreResult.score}/100
              </div>
              <p className="mt-2 text-sm text-blue-800">{scoreResult.reasoning}</p>
            </div>
            <RunStatistics stats={scoreResult} />
          </div>
        )}
      </div> */}

      {/* <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <h2 className="text-xl font-semibold text-gray-900">Score a Set of Questions</h2>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">User Need</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
            value={setNeedText}
            onChange={(e) => setSetNeedText(e.target.value)}
            placeholder="Describe the user requirement..."
            disabled={setScoreBusy}
          />
        </div>
        <FileUpload
          accept=".csv"
          file={setCsvFile}
          onChange={setSetCsvFile}
          label="Upload CSV (generated questions)"
          disabled={setScoreBusy}
        />
        <Input
          label="Question Column"
          value={setTextColumn}
          onChange={(e) => setSetTextColumn(e.target.value)}
          placeholder="question"
          disabled={setScoreBusy}
        />
        {setScoreErrorMsg && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {setScoreErrorMsg}
          </div>
        )}
        <Button onClick={handleSetScore} loading={setScoreBusy} disabled={setScoreBusy}>
          Score Question Set
        </Button>

        {setScoreData && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-700">Set Score</p>
              <div className="text-4xl font-bold text-blue-900">
                {setScoreData.score}/100
              </div>
              <p className="mt-2 text-sm text-blue-800">{setScoreData.reasoning}</p>
            </div>
            <RunStatistics stats={setScoreData} />
          </div>
        )}
      </div> */}

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <h2 className="text-xl font-semibold text-gray-900">Score a Set of Questions (CSV)</h2>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">User Need</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
            value={setFileNeedText}
            onChange={(e) => setSetFileNeedText(e.target.value)}
            placeholder="Describe the user requirement..."
            disabled={setFileScoreBusy}
          />
        </div>
        <FileUpload
          accept=".csv"
          file={setFileCsv}
          onChange={setSetFileCsv}
          label="Upload CSV (generated questions)"
          disabled={setFileScoreBusy}
        />
        <Input
          label="Question Column"
          value={setFileTextColumn}
          onChange={(e) => setSetFileTextColumn(e.target.value)}
          placeholder="question"
          disabled={setFileScoreBusy}
        />
        {setFileScoreErrorMsg && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {setFileScoreErrorMsg}
          </div>
        )}
        <Button onClick={handleSetScoreFile} loading={setFileScoreBusy} disabled={setFileScoreBusy}>
          Score Question Set (File)
        </Button>

        {setFileScoreData && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-700">Set Score</p>
              <div className="text-4xl font-bold text-blue-900">
                {setFileScoreData.score}/100
              </div>
              <p className="mt-2 text-sm text-blue-800">{setFileScoreData.reasoning}</p>
              <p className="mt-2 text-sm text-blue-800">
                Questions scored: {setFileScoreData.question_count}
              </p>
            </div>
            <RunStatistics stats={setFileScoreData} />
          </div>
        )}
      </div>

      {/* <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <h2 className="text-xl font-semibold text-gray-900">Compare Two Questions</h2>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">User Need</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
            value={compareNeed}
            onChange={(e) => setCompareNeed(e.target.value)}
            placeholder="Describe the user requirement..."
            disabled={compareLoading}
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Question A</label>
            <textarea
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={4}
              value={questionA}
              onChange={(e) => setQuestionA(e.target.value)}
              placeholder="Question A..."
              disabled={compareLoading}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Question B</label>
            <textarea
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={4}
              value={questionB}
              onChange={(e) => setQuestionB(e.target.value)}
              placeholder="Question B..."
              disabled={compareLoading}
            />
          </div>
        </div>
        {compareError && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {compareError}
          </div>
        )}
        <Button onClick={handleCompare} loading={compareLoading} disabled={compareLoading}>
          Compare Questions
        </Button>

        {compareResult && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-emerald-50 rounded-lg space-y-2">
              <p className="text-sm text-emerald-700">Winner</p>
              <div className="text-3xl font-bold text-emerald-900">
                {renderWinner(compareResult.winner)}
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm text-emerald-800">
                <div>Question A score: {compareResult.score_a}/100</div>
                <div>Question B score: {compareResult.score_b}/100</div>
              </div>
              <p className="text-sm text-emerald-800">{compareResult.reasoning}</p>
            </div>
            <RunStatistics stats={compareResult} />
          </div>
        )}
      </div> */}

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <h2 className="text-xl font-semibold text-gray-900">Compare Two Question Sets</h2>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">User Need</label>
          <textarea
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={4}
            value={setCompareNeedText}
            onChange={(e) => setSetCompareNeedText(e.target.value)}
            placeholder="Describe the user requirement..."
            disabled={setCompareBusy}
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <FileUpload
            accept=".csv"
            file={setCompareFileA}
            onChange={setSetCompareFileA}
            label="Upload CSV (Set A)"
            disabled={setCompareBusy}
          />
          <FileUpload
            accept=".csv"
            file={setCompareFileB}
            onChange={setSetCompareFileB}
            label="Upload CSV (Set B)"
            disabled={setCompareBusy}
          />
        </div>
        <Input
          label="Question Column"
          value={setCompareTextColumn}
          onChange={(e) => setSetCompareTextColumn(e.target.value)}
          placeholder="question"
          disabled={setCompareBusy}
        />
        {setCompareErrorMsg && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {setCompareErrorMsg}
          </div>
        )}
        <Button onClick={handleSetCompare} loading={setCompareBusy} disabled={setCompareBusy}>
          Compare Question Sets
        </Button>

        {setCompareData && (
          <div className="mt-4 space-y-4">
            <div className="p-4 bg-emerald-50 rounded-lg space-y-2">
              <p className="text-sm text-emerald-700">Winner</p>
              <div className="text-3xl font-bold text-emerald-900">
                {renderSetWinner(setCompareData.winner)}
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm text-emerald-800">
                <div>Set A score: {setCompareData.score_a}/100</div>
                <div>Set B score: {setCompareData.score_b}/100</div>
              </div>
              <p className="text-sm text-emerald-800">{setCompareData.reasoning}</p>
            </div>
            <RunStatistics stats={setCompareData} />
          </div>
        )}
      </div>
    </div>
  );
}
