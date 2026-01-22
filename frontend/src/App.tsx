import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Home, FileText, Brain, TrendingUp } from 'lucide-react';
import WorkflowPage from './pages/WorkflowPage';
import KNNPredictionPage from './pages/KNNPredictionPage';
import LLMLabelingPage from './pages/LLMLabelingPage';
import EvaluationPage from './pages/EvaluationPage';

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
                    Semi-Supervised Labeling Framework
                  </span>
                </div>
                <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                  <NavLink to="/" icon={<Home className="h-5 w-5" />}>
                    Workflow
                  </NavLink>
                  <NavLink to="/knn-prediction" icon={<Brain className="h-5 w-5" />}>
                    KNN Prediction
                  </NavLink>
                  <NavLink to="/llm-labeling" icon={<FileText className="h-5 w-5" />}>
                    LLM Labeling
                  </NavLink>
                  <NavLink to="/evaluation" icon={<TrendingUp className="h-5 w-5" />}>
                    Evaluation
                  </NavLink>
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

function NavLink({ to, icon, children }: NavLinkProps) {
  return (
    <Link
      to={to}
      className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:text-gray-700 hover:border-gray-300 transition-colors"
    >
      <span className="mr-2">{icon}</span>
      {children}
    </Link>
  );
}

export default App;
