import { BrowserRouter as Router, Routes, Route, NavLink as RouterNavLink } from 'react-router-dom';
import { Home, FileText, Brain, TrendingUp, Search, CheckCircle } from 'lucide-react';
import WorkflowPage from './pages/WorkflowPage';
import KNNPredictionPage from './pages/KNNPredictionPage';
import LLMLabelingPage from './pages/LLMLabelingPage';
import EvaluationPage from './pages/EvaluationPage';
import SelectQuestionPage from './pages/SelectQuestionPage';
import QuestionJudgePage from './pages/QuestionJudgePage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <nav className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex">
                <div className="flex-shrink-0 flex items-center">
                  <Brain className="h-8 w-8 text-blue-600" />
                  <span className="ml-2 text-xl font-bold text-gray-900">
                    SSSL
                  </span>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <AppNavLink to="/" icon={<Home className="h-5 w-5" />}>
                    Workflow
                  </AppNavLink>
                  <AppNavLink to="/knn-prediction" icon={<Brain className="h-5 w-5" />}>
                    KNN Prediction
                  </AppNavLink>
                  <AppNavLink to="/llm-labeling" icon={<FileText className="h-5 w-5" />}>
                    LLM Labeling
                  </AppNavLink>
                  <AppNavLink to="/evaluation" icon={<TrendingUp className="h-5 w-5" />}>
                    Evaluation
                  </AppNavLink>
                  <AppNavLink to="/select-questions" icon={<Search className="h-5 w-5" />}>
                    Select Questions
                  </AppNavLink>
                  <AppNavLink to="/question-judge" icon={<CheckCircle className="h-5 w-5" />}>
                    Question Judge
                  </AppNavLink>
                </div>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <Routes>
            <Route path="/" element={<WorkflowPage />} />
            <Route path="/knn-prediction" element={<KNNPredictionPage />} />
            <Route path="/llm-labeling" element={<LLMLabelingPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/select-questions" element={<SelectQuestionPage />} />
            <Route path="/question-judge" element={<QuestionJudgePage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

interface NavLinkProps {
  to: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

function AppNavLink({ to, icon, children }: NavLinkProps) {
  return (
    <RouterNavLink
      to={to}
      className={({ isActive }) =>
        `inline-flex items-center px-3 pt-1 border-b-2 text-sm font-medium transition-colors ${
          isActive
            ? 'border-blue-600 text-blue-700 bg-blue-50 rounded-t-md'
            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
        }`
      }
      end={to === '/'}
    >
      <span className="mr-2">{icon}</span>
      {children}
    </RouterNavLink>
  );
}

export default App;
